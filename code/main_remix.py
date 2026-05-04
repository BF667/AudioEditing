import argparse
import calendar
import os
import time

import matplotlib.pyplot as plt
import torch
import torchaudio
import wandb
from torch import inference_mode

from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from models import load_model
from utils import set_reproducability, load_audio, get_spec


HF_TOKEN = None  # Needed for stable audio open. You can leave None when not using it


def remix_style_transfer(args, device):
    """Remix by transferring the style of one audio onto the structure of another.

    Inverts source A into latent space, then generates using a prompt that describes
    the style of source B. The result keeps the structure/arrangement of A but sounds
    like the style described in the prompt.
    """
    ldm_stable = load_model(args.model_id, device, args.num_diffusion_steps, token=HF_TOKEN)

    # Load source A (the audio whose structure we keep)
    x0_a, sr, duration = load_audio(
        args.source_a, ldm_stable.get_fn_STFT(), device=device,
        stft=('stable-audio' not in args.model_id), model_sr=ldm_stable.get_sr())
    torch.cuda.empty_cache()

    with inference_mode():
        w0 = ldm_stable.vae_encode(x0_a)

    # Forward inversion of source A
    print("[1/3] Inverting source audio...")
    with inference_mode():
        wt, zs, wts, extra_info = inversion_forward_process(
            ldm_stable, w0, etas=1.0,
            prompts=args.source_prompt,
            cfg_scales=[args.cfg_src],
            prog_bar=True,
            num_inference_steps=args.num_diffusion_steps,
            numerical_fix=True,
            duration=duration)

    # Reverse with target style prompt
    print("[2/3] Generating remixed audio...")
    tstart = torch.tensor([args.tstart], dtype=torch.int)
    skip = args.num_diffusion_steps - tstart

    with inference_mode():
        w0_remix, _ = inversion_reverse_process(
            ldm_stable, xT=wts, tstart=tstart,
            fix_alpha=args.fix_alpha, etas=1.0,
            prompts=args.target_prompt,
            neg_prompts=args.target_neg_prompt,
            cfg_scales=[args.cfg_tar], prog_bar=True,
            zs=zs[:int(args.num_diffusion_steps - min(skip))],
            duration=duration, extra_info=extra_info)

    return ldm_stable, w0_remix, w0, sr, duration


def remix_latent_blend(args, device):
    """Remix by blending the latent encodings of two audio sources.

    Encodes both sources into the latent space, blends them with a controllable ratio,
    then decodes the blended representation. Higher blend_ratio = more of source B.
    """
    ldm_stable = load_model(args.model_id, device, args.num_diffusion_steps, token=HF_TOKEN)

    # Load both sources
    x0_a, sr, duration_a = load_audio(
        args.source_a, ldm_stable.get_fn_STFT(), device=device,
        stft=('stable-audio' not in args.model_id), model_sr=ldm_stable.get_sr())

    x0_b, _, duration_b = load_audio(
        args.source_b, ldm_stable.get_fn_STFT(), device=device,
        stft=('stable-audio' not in args.model_id), model_sr=ldm_stable.get_sr())
    torch.cuda.empty_cache()

    duration = min(duration_a, duration_b)

    with inference_mode():
        w0_a = ldm_stable.vae_encode(x0_a)
        w0_b = ldm_stable.vae_encode(x0_b)

    # Trim to same length if needed
    min_len = min(w0_a.shape[-1], w0_b.shape[-1])
    w0_a = w0_a[:, :, :, :min_len]
    w0_b = w0_b[:, :, :, :min_len]

    # Blend latents
    print(f"[1/2] Blending latents (ratio={args.blend_ratio:.2f} = {args.blend_ratio:.0%} source B)...")
    blend = (1 - args.blend_ratio) * w0_a + args.blend_ratio * w0_b

    # Forward inversion of blended latent
    print("[2/2] Inverting and reconstructing blended audio...")
    with inference_mode():
        wt, zs, wts, extra_info = inversion_forward_process(
            ldm_stable, blend, etas=1.0,
            prompts=args.source_prompt,
            cfg_scales=[args.cfg_src],
            prog_bar=True,
            num_inference_steps=args.num_diffusion_steps,
            numerical_fix=True,
            duration=duration)

    # Full reverse to get clean output
    tstart = torch.tensor([0], dtype=torch.int)
    with inference_mode():
        w0_remix, _ = inversion_reverse_process(
            ldm_stable, xT=wts, tstart=tstart,
            fix_alpha=0.0, etas=1.0,
            prompts=args.source_prompt,
            neg_prompts=[""],
            cfg_scales=[args.cfg_src], prog_bar=True,
            zs=zs, duration=duration, extra_info=extra_info)

    return ldm_stable, w0_remix, w0_a, sr, duration


def remix_temporal(args, device):
    """Remix by combining temporal segments from two sources.

    Uses the inversion process to seamlessly transition between two audio sources.
    The first portion uses source A's characteristics, transitioning to source B's
    style via a target prompt at the specified timestep.
    """
    ldm_stable = load_model(args.model_id, device, args.num_diffusion_steps, token=HF_TOKEN)

    # Load source A
    x0_a, sr, duration = load_audio(
        args.source_a, ldm_stable.get_fn_STFT(), device=device,
        stft=('stable-audio' not in args.model_id), model_sr=ldm_stable.get_sr())
    torch.cuda.empty_cache()

    with inference_mode():
        w0 = ldm_stable.vae_encode(x0_a)

    # Build dual prompts for temporal segmentation
    # The first prompt edits the first portion, the second edits the second portion
    if args.source_b_prompt and args.target_prompt[0] != "":
        prompts = [args.source_prompt[0] if args.source_prompt else "",
                   args.target_prompt[0]]
        cfg_scales = [args.cfg_src, args.cfg_tar]
    else:
        prompts = args.target_prompt
        cfg_scales = [args.cfg_tar]

    # Forward inversion of source A
    print("[1/3] Inverting source audio...")
    with inference_mode():
        wt, zs, wts, extra_info = inversion_forward_process(
            ldm_stable, w0, etas=1.0,
            prompts=prompts,
            cfg_scales=cfg_scales,
            prog_bar=True,
            num_inference_steps=args.num_diffusion_steps,
            cutoff_points=[args.mix_point] if len(prompts) > 1 else None,
            numerical_fix=True,
            duration=duration)

    # Reverse with temporal blend
    print("[2/3] Generating temporally remixed audio...")
    tstart = torch.tensor([args.tstart], dtype=torch.int)
    skip = args.num_diffusion_steps - tstart

    with inference_mode():
        w0_remix, _ = inversion_reverse_process(
            ldm_stable, xT=wts, tstart=tstart,
            fix_alpha=args.fix_alpha, etas=1.0,
            prompts=prompts,
            neg_prompts=args.target_neg_prompt * len(prompts) if args.target_neg_prompt else [""],
            cfg_scales=cfg_scales, prog_bar=True,
            zs=zs[:int(args.num_diffusion_steps - min(skip))],
            cutoff_points=[args.mix_point] if len(prompts) > 1 else None,
            duration=duration, extra_info=extra_info)

    return ldm_stable, w0_remix, w0, sr, duration


def save_results(ldm_stable, w0_remix, w0_orig, sr, args):
    """Decode and save the remixed audio."""
    print("[3/3] Decoding and saving...")
    with inference_mode():
        x0_dec = ldm_stable.vae_decode(w0_remix)
        if 'stable-audio' not in args.model_id:
            if x0_dec.dim() < 4:
                x0_dec = x0_dec[None, :, :, :]
            with torch.no_grad():
                audio = ldm_stable.decode_to_mel(x0_dec)
                orig_audio = ldm_stable.decode_to_mel(x0_orig)
        else:
            audio = x0_dec.detach().clone().cpu().squeeze(0)
            orig_audio = w0_orig.detach().clone().cpu()
            x0_dec = get_spec(x0_dec, ldm_stable.get_fn_STFT())
            x0_orig_spec = get_spec(w0_orig.unsqueeze(0), ldm_stable.get_fn_STFT())
            if x0_dec.dim() < 4:
                x0_dec = x0_dec[None, :, :, :]
                x0_orig_spec = x0_orig_spec[None, :, :, :]
            orig_audio = x0_orig_spec  # skip for spec comparison

    # Build save path
    current_gmt = time.gmtime()
    ts = calendar.timegm(current_gmt)
    mode_str = args.mode
    if args.mode == 'latent_blend':
        mode_str = f'blend{args.blend_ratio:.1f}'
    elif args.mode == 'temporal':
        mode_str = f'temporal_mix{args.mix_point:.1f}'

    image_name = f'remix_{mode_str}_skip{args.num_diffusion_steps - args.tstart}_cfg{args.cfg_tar}_{ts}'

    save_path = os.path.join(
        args.results_path,
        args.model_id.split('/')[1],
        f'{os.path.basename(args.source_a).split(".")[0]}'
        f'{"_" + os.path.basename(args.source_b).split(".")[0] if args.source_b else ""}',
        mode_str)
    os.makedirs(save_path, exist_ok=True)

    # Save spectrogram and audio
    if x0_dec.shape[2] > x0_dec.shape[3]:
        spec = x0_dec[0, 0].T.cpu().detach().numpy()
    else:
        spec = x0_dec[0, 0].cpu().detach().numpy()

    plt.imsave(os.path.join(save_path, image_name + ".png"), spec)

    # Determine output format from source extension or args
    out_ext = ".wav"
    if hasattr(args, 'output_format') and args.output_format:
        out_ext = f".{args.output_format}"
    elif args.source_a and args.source_a.lower().endswith('.mp3'):
        out_ext = ".mp3"

    torchaudio.save(os.path.join(save_path, image_name + out_ext), audio, sample_rate=sr, format=out_ext[1:])
    torchaudio.save(os.path.join(save_path, "orig" + out_ext), orig_audio, sample_rate=sr, format=out_ext[1:])

    # Save source B if provided
    if args.source_b and os.path.exists(args.source_b):
        wav_b, sr_b = torchaudio.load(args.source_b)
        if sr_b != sr:
            wav_b = torchaudio.functional.resample(wav_b, orig_freq=sr_b, new_freq=sr)
        torchaudio.save(os.path.join(save_path, "source_b" + out_ext), wav_b, sample_rate=sr, format=out_ext[1:])

    print(f"\nResults saved to: {save_path}")
    print(f"  - {image_name}{out_ext}  (remix)")
    print(f"  - orig{out_ext}  (source A)")
    if args.source_b:
        print(f"  - source_b{out_ext}  (source B)")

    # Wandb logging
    if not args.wandb_disable:
        log_dict = {
            'orig': wandb.Audio(orig_audio.squeeze(), caption='Source A', sample_rate=sr),
            'remix': wandb.Audio(audio[0].squeeze(), caption=image_name, sample_rate=sr),
            'remix_spec': wandb.Image(spec, caption=image_name),
        }
        wandb.log(log_dict)

    return save_path, image_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Music Remixing using DDPM Inversion')

    # Mode
    parser.add_argument("--mode", type=str, default="style_transfer",
                        choices=["style_transfer", "latent_blend", "temporal"],
                        help="Remix mode: style_transfer, latent_blend, or temporal")

    # Model
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument('-s', "--seed", type=int, default=None)
    parser.add_argument("--model_id", type=str,
                        choices=["cvssp/audioldm-s-full-v2",
                                 "cvssp/audioldm-l-full",
                                 "cvssp/audioldm2",
                                 "cvssp/audioldm2-large",
                                 "cvssp/audioldm2-music",
                                 'declare-lab/tango-full-ft-audio-music-caps',
                                 'declare-lab/tango-full-ft-audiocaps',
                                 "stabilityai/stable-audio-open-1.0"],
                        default="cvssp/audioldm2-music")

    # Audio sources
    parser.add_argument("--source_a", type=str, required=True,
                        help='Primary audio source (provides structure/content)')
    parser.add_argument("--source_b", type=str, default=None,
                        help='Secondary audio source (for latent_blend mode)')

    # Text prompts
    parser.add_argument("--source_prompt", type=str, nargs='+', default=[""],
                        help='Prompt describing the original audio (source A)')
    parser.add_argument("--target_prompt", type=str, nargs='+', default=[""],
                        help='Prompt describing the desired remix style')
    parser.add_argument("--source_b_prompt", type=str, default=None,
                        help='Prompt describing source B (for temporal mode)')

    # Editing parameters
    parser.add_argument("--cfg_src", type=float, nargs='+', default=[3],
                        help='CFG strength for forward (inversion) process')
    parser.add_argument("--cfg_tar", type=float, nargs='+', default=[12],
                        help='CFG strength for reverse (generation) process')
    parser.add_argument("--num_diffusion_steps", type=int, default=200)
    parser.add_argument("--tstart", type=int, default=100,
                        help='Timestep to start reverse process (editing strength)')
    parser.add_argument("--fix_alpha", type=float, default=0.1,
                        help='Mask fix strength for smoother transitions')
    parser.add_argument("--target_neg_prompt", type=str, nargs='+', default=[""],
                        help='Negative prompt for reverse process')

    # Blend-specific parameters
    parser.add_argument("--blend_ratio", type=float, default=0.5,
                        help='Blend ratio for latent_blend mode (0=all A, 1=all B)')

    # Temporal-specific parameters
    parser.add_argument("--mix_point", type=float, default=0.5,
                        help='Temporal mix point (0-1, fraction of audio for source A)')

    # Output
    parser.add_argument("--results_path", type=str, default="remix_results")
    parser.add_argument("--output_format", type=str, default="wav",
                        choices=["wav", "mp3"],
                        help='Output audio format (default: same as source, or wav)')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_disable', action='store_true', default=True)

    args = parser.parse_args()

    if args.model_id == "stabilityai/stable-audio-open-1.0" and HF_TOKEN is None:
        raise ValueError("HF_TOKEN is required for stable audio model. Set it in main_remix.py:HF_TOKEN")

    if args.mode == "latent_blend" and args.source_b is None:
        raise ValueError("--source_b is required for latent_blend mode")

    set_reproducability(args.seed, extreme=False)
    device = f"cuda:{args.device_num}"
    torch.cuda.set_device(args.device_num)

    # Init wandb
    wandb.login(key='')
    wandb_run = wandb.init(
        project="AudInv", entity='', config={},
        name=args.wandb_name if args.wandb_name else f'remix_{args.mode}',
        group=args.wandb_group,
        mode='disabled' if args.wandb_disable else 'online',
        settings=wandb.Settings(_disable_stats=True))
    wandb.config.update(args)

    # Run selected remix mode
    print(f"\n{'='*50}")
    print(f"  Remix Mode: {args.mode}")
    print(f"  Source A: {os.path.basename(args.source_a)}")
    if args.source_b:
        print(f"  Source B: {os.path.basename(args.source_b)}")
    if args.target_prompt[0]:
        print(f"  Target prompt: {args.target_prompt[0]}")
    print(f"{'='*50}\n")

    if args.mode == "style_transfer":
        ldm_stable, w0_remix, w0_orig, sr, duration = remix_style_transfer(args, device)
    elif args.mode == "latent_blend":
        ldm_stable, w0_remix, w0_orig, sr, duration = remix_latent_blend(args, device)
    elif args.mode == "temporal":
        ldm_stable, w0_remix, w0_orig, sr, duration = remix_temporal(args, device)

    save_results(ldm_stable, w0_remix, w0_orig, sr, args)
    wandb_run.finish()
    print("\nDone!")
