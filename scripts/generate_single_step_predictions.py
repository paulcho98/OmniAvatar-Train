"""Generate single-step x0_pred at each timestep, from fresh noise or Euler-jumped noise.

Two modes:
  --mode fresh_noise:  x_t = (1-t)*x_0 + t*eps  (eps ~ N(0,I), fixed seed)
                       Simulates what the model sees during training at each noise level.

  --mode euler_jump:   x_t = x_t_0 + (t - t_0) * v_0  (Euler from step 0's prediction)
                       Tests trajectory straightness: can a single Euler jump from step 0
                       produce an x_t that the teacher handles well?

For each timestep, runs the teacher with CFG → x0_pred, decodes via VAE → saves mp4.
Output is compatible with eval_ode_perceptual_v2.py --phase metrics.

Usage:
    # Fresh noise (GPU 0)
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_single_step_predictions.py \
        --mode fresh_noise \
        --traj_dir /home/work/ode_full_trajectories/14B \
        --output_dir /home/work/.local/ode_analysis/14B/fresh_noise \
        --samples "1ec7a45d803ab472e7fe5c6625667289_shot_001_000,17ef723a912e46713e84fc2b7dd74e23_shot_001_000"

    # Euler jump (GPU 1)
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_single_step_predictions.py \
        --mode euler_jump \
        --traj_dir /home/work/ode_full_trajectories/14B \
        --output_dir /home/work/.local/ode_analysis/14B/euler_jump \
        --samples "1ec7a45d803ab472e7fe5c6625667289_shot_001_000,17ef723a912e46713e84fc2b7dd74e23_shot_001_000"
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/home/work/.local/hyunbin/FastGen")

from fastgen.networks.OmniAvatar.network import OmniAvatarWan
from OmniAvatar.models.model_manager import ModelManager


# ── Paths ──
PRETRAINED = "/home/work/.local/OmniAvatar/pretrained_models"
BASE_14B = ",".join([
    f"{PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-{i:05d}-of-00006.safetensors"
    for i in range(1, 7)
])
CKPT_14B = "/home/work/output_omniavatar_v2v_phase2/step-10500.pt"
VAE_PATH = f"{PRETRAINED}/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
MASK_PATH = "/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png"
DATA_DIR = "/home/work/stableavatar_data/v2v_validation_data/recon"
NEG_TEXT_EMB = "/home/work/stableavatar_data/neg_text_emb.pt"
AUDIO_BASE_DIR = DATA_DIR  # audio.wav is in each sample dir


def load_mask(latent_h=64, latent_w=64):
    mask_arr = np.array(Image.open(MASK_PATH), dtype=np.float32)[:, :, 0] / 255.0
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_t, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
    return (mask_resized.squeeze() > 0.5).float()


def load_condition(sample_name, mask, neg_text_embeds, device, dtype, cfg_drop_text=True):
    """Load condition dict for a sample."""
    sample_dir = os.path.join(DATA_DIR, sample_name)

    vae_path = os.path.join(sample_dir, "vae_latents_mask_all.pt")
    if not os.path.exists(vae_path):
        vae_path = os.path.join(sample_dir, "vae_latents.pt")
    vae_data = torch.load(vae_path, map_location="cpu", weights_only=False)
    input_latents = vae_data["input_latents"].to(dtype)
    masked_latents = vae_data["masked_latents"].to(dtype)

    audio_data = torch.load(os.path.join(sample_dir, "audio_emb_omniavatar.pt"),
                            map_location="cpu", weights_only=False)
    audio_emb = audio_data["audio_emb"][:81].to(dtype)

    text_emb = torch.load(os.path.join(sample_dir, "text_emb.pt"),
                          map_location="cpu", weights_only=False)
    if isinstance(text_emb, dict):
        text_emb = next(v for v in text_emb.values() if isinstance(v, torch.Tensor))
    text_emb = text_emb.to(dtype)
    if text_emb.dim() == 2:
        text_emb = text_emb.unsqueeze(0)

    ref_latent = input_latents[:, 0:1, :, :]

    ref_seq = None
    ref_path = os.path.join(sample_dir, "ref_latents.pt")
    if os.path.exists(ref_path):
        ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
        ref_seq = ref_data["ref_sequence_latents"].to(dtype)

    condition = {
        "text_embeds": text_emb.unsqueeze(0).to(device),
        "audio_emb": audio_emb.unsqueeze(0).to(device),
        "ref_latent": ref_latent.unsqueeze(0).to(device),
        "mask": mask.to(device),
        "masked_video": masked_latents.unsqueeze(0).to(device),
        "ref_sequence": (ref_seq.unsqueeze(0).to(device) if ref_seq is not None
                         else torch.zeros_like(masked_latents.unsqueeze(0)).to(device)),
    }
    if condition["text_embeds"].dim() == 4:
        condition["text_embeds"] = condition["text_embeds"].squeeze(1)

    if cfg_drop_text:
        neg_text = neg_text_embeds.to(device=device, dtype=dtype)
        if neg_text.dim() == 2:
            neg_text = neg_text.unsqueeze(0)
    else:
        # audio-only CFG: keep positive text
        neg_text = condition["text_embeds"]
    neg_condition = {
        "text_embeds": neg_text,
        "audio_emb": torch.zeros_like(condition["audio_emb"]),
        "ref_latent": condition["ref_latent"],
        "mask": condition["mask"],
        "masked_video": condition["masked_video"],
        "ref_sequence": condition["ref_sequence"],
    }

    return condition, neg_condition, input_latents


def teacher_predict_x0(teacher, x_t, t_val, condition, neg_condition, guidance_scale=4.5):
    t_tensor = torch.tensor([t_val], device=x_t.device, dtype=torch.float32)
    x0_cond = teacher(x_t, t_tensor, condition=condition, fwd_pred_type="x0")
    if guidance_scale != 1.0:
        x0_uncond = teacher(x_t, t_tensor, condition=neg_condition, fwd_pred_type="x0")
        x0_pred = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
    else:
        x0_pred = x0_cond
    return x0_pred


def decode_latents(vae, latents, device):
    latents = latents.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        frames = vae.decode(latents, device=device, tiled=False)
    if frames.dim() == 5:
        frames = frames[0]
    frames = ((frames.float() + 1) / 2 * 255).clamp(0, 255).byte()
    return frames.permute(1, 2, 3, 0).cpu().numpy()


def save_video(frames_np, path, fps=25):
    import imageio
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()


def mux_audio(video_path, audio_path, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path, "-i", audio_path,
        "-c:v", "libx264", "-crf", "18", "-c:a", "aac",
        "-shortest", output_path,
    ], capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["fresh_noise", "euler_jump"])
    parser.add_argument("--traj_dir", type=str, required=True,
                        help="Directory with saved ODE trajectories (for schedule + Euler data)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--samples", type=str, required=True,
                        help="Comma-separated sample names")
    parser.add_argument("--guidance_scale", type=float, default=4.5,
                        help="CFG scale for fresh_noise mode (both step 0 and teacher)")
    parser.add_argument("--cfg_step0", type=float, default=None,
                        help="CFG scale for step 0 prediction in euler_jump mode "
                             "(default: same as --guidance_scale). Set to 1.0 for no CFG.")
    parser.add_argument("--cfg_teacher", type=float, default=None,
                        help="CFG scale for teacher re-evaluation in euler_jump mode "
                             "(default: same as --guidance_scale). Set to 1.0 for no CFG.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--cfg_drop_text", type=str, default="true",
                        choices=["true", "false"],
                        help="If true (default), negative branch uses negative text embedding. "
                             "If false, negative branch keeps positive text embedding (audio-only CFG).")
    args = parser.parse_args()

    # Default CFG params
    if args.cfg_step0 is None:
        args.cfg_step0 = args.guidance_scale
    if args.cfg_teacher is None:
        args.cfg_teacher = args.guidance_scale

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    sample_names = args.samples.split(",")

    # Load schedule
    with open(os.path.join(args.traj_dir, sample_names[0], "ode_schedule.json")) as f:
        schedule = json.load(f)
    t_list = schedule["t_list"]
    num_steps = schedule["num_steps"]

    cfg_mode = "text+audio (original)" if args.cfg_drop_text == "true" else "audio-only"
    print(f"CFG drop mode: {cfg_mode}")
    print(f"Mode: {args.mode}")
    print(f"Samples: {len(sample_names)}")
    print(f"Steps: {num_steps}, t range: [{t_list[0]:.4f}, {t_list[-2]:.4f}]")

    # Load teacher
    print("Loading 14B teacher...")
    teacher = OmniAvatarWan(
        model_size="14B", in_dim=65, mode="v2v", use_audio=True,
        base_model_paths=BASE_14B, omniavatar_ckpt_path=CKPT_14B,
        merge_lora=True, net_pred_type="flow", schedule_type="rf",
    ).to(device, dtype=dtype).eval()
    teacher.requires_grad_(False)
    print("Teacher loaded.")

    # Load VAE
    print("Loading VAE...")
    model_manager = ModelManager(device="cpu", infer=True)
    model_manager.load_models([VAE_PATH], torch_dtype=torch.bfloat16, device="cpu")
    vae_idx = model_manager.model_name.index("wan_video_vae")
    vae = model_manager.model[vae_idx].to(device)

    # Load mask and neg text
    mask = load_mask()
    neg_text_embeds = torch.load(NEG_TEXT_EMB, map_location="cpu", weights_only=False)
    if isinstance(neg_text_embeds, dict):
        neg_text_embeds = next(v for v in neg_text_embeds.values() if isinstance(v, torch.Tensor))
    neg_text_embeds = neg_text_embeds.to(dtype)

    videos_dir = os.path.join(args.output_dir, "videos")
    t_start = time.time()

    for si, sample_name in enumerate(sample_names):
        print(f"\n=== [{si+1}/{len(sample_names)}] {sample_name} ===")
        sample_out = os.path.join(videos_dir, sample_name)
        os.makedirs(sample_out, exist_ok=True)

        # Load conditions
        condition, neg_condition, gt_latents_raw = load_condition(
            sample_name, mask, neg_text_embeds, device, dtype,
            cfg_drop_text=(args.cfg_drop_text == "true"),
        )

        # GT latents (float32 for noise construction)
        gt_latents = torch.load(
            os.path.join(args.traj_dir, sample_name, "input_latents.pt"),
            map_location="cpu", weights_only=True,
        ).float()

        audio_path = os.path.join(AUDIO_BASE_DIR, sample_name, "audio.wav")
        has_audio = os.path.exists(audio_path)

        # Decode + save GT once
        gt_vid_path = os.path.join(sample_out, "gt.mp4")
        if not os.path.exists(gt_vid_path):
            print("  Decoding GT...")
            gt_frames = decode_latents(vae, gt_latents, device)
            save_video(gt_frames, gt_vid_path)
            if has_audio:
                mux_audio(gt_vid_path, audio_path, gt_vid_path.replace(".mp4", "_audio.mp4"))

        # Mode-specific setup
        if args.mode == "fresh_noise":
            # Sample fixed noise
            torch.manual_seed(args.seed)
            eps = torch.randn_like(gt_latents)  # [16, 21, 64, 64]
            print(f"  Fresh noise mode (seed={args.seed})")

        elif args.mode == "euler_jump":
            # Load step 0's x_t from trajectory (this is the same regardless of CFG)
            x_t_0 = torch.load(
                os.path.join(args.traj_dir, sample_name, "step_000_xt.pt"),
                map_location="cpu", weights_only=True,
            ).float()
            t_0 = t_list[0]

            # Get x0_pred at step 0 with the specified CFG
            if args.cfg_step0 == 4.5:
                # Same as trajectory — load saved prediction
                x0_0 = torch.load(
                    os.path.join(args.traj_dir, sample_name, "step_000_x0.pt"),
                    map_location="cpu", weights_only=True,
                ).float()
                print(f"  Euler jump mode (step0 CFG={args.cfg_step0}, "
                      f"teacher CFG={args.cfg_teacher}, t_0={t_0:.4f}) [saved x0]")
            else:
                # Re-run teacher at step 0 with different CFG
                print(f"  Re-running step 0 with CFG={args.cfg_step0}...")
                x_t_0_gpu = x_t_0.unsqueeze(0).to(device=device, dtype=dtype)
                with torch.no_grad():
                    x0_0_gpu = teacher_predict_x0(
                        teacher, x_t_0_gpu, t_0, condition, neg_condition, args.cfg_step0
                    )
                x0_0 = x0_0_gpu.squeeze(0).float().cpu()
                del x_t_0_gpu, x0_0_gpu
                torch.cuda.empty_cache()
                print(f"  Euler jump mode (step0 CFG={args.cfg_step0}, "
                      f"teacher CFG={args.cfg_teacher}, t_0={t_0:.4f}) [recomputed x0]")

            # Recover noise direction from step 0
            eps_euler = (x_t_0 - (1.0 - t_0) * x0_0) / t_0

        # Generate predictions at each timestep
        for step_i in range(num_steps):
            t_val = t_list[step_i]
            step_path = os.path.join(sample_out, f"step_{step_i:03d}.mp4")

            if args.skip_existing and os.path.exists(step_path):
                if step_i % 10 == 0:
                    print(f"  step {step_i} (t={t_val:.3f}) — skipped (exists)")
                continue

            # Construct x_t
            if args.mode == "fresh_noise":
                # RF interpolation: x_t = (1-t)*x_0 + t*eps
                x_t = ((1.0 - t_val) * gt_latents + t_val * eps)
            elif args.mode == "euler_jump":
                # Euler jump from step 0
                if t_val > 1e-6:
                    x_t = (1.0 - t_val) * x0_0 + t_val * eps_euler
                else:
                    x_t = x0_0

            x_t_gpu = x_t.unsqueeze(0).to(device=device, dtype=dtype)

            # Teacher prediction (use cfg_teacher for euler_jump, guidance_scale for fresh_noise)
            cfg = args.cfg_teacher if args.mode == "euler_jump" else args.guidance_scale
            with torch.no_grad():
                x0_pred = teacher_predict_x0(
                    teacher, x_t_gpu, t_val, condition, neg_condition, cfg
                )

            # Decode
            x0_np = decode_latents(vae, x0_pred.squeeze(0).float().cpu(), device)
            save_video(x0_np, step_path)

            if has_audio:
                mux_audio(step_path, audio_path, step_path.replace(".mp4", "_audio.mp4"))

            elapsed = time.time() - t_start
            if step_i % 10 == 0 or step_i == num_steps - 1:
                print(f"  step {step_i:2d} (t={t_val:.3f}) | elapsed {elapsed/60:.1f} min")

            # Cleanup
            del x_t_gpu, x0_pred
            torch.cuda.empty_cache()

    elapsed_total = time.time() - t_start
    print(f"\nDone! {len(sample_names)} samples × {num_steps} steps in {elapsed_total/60:.1f} min")
    print(f"Videos: {videos_dir}")
    print(f"\nTo run metrics:")
    print(f"  python scripts/eval_ode_perceptual_v2.py --phase metrics \\")
    print(f"    --traj_dir {args.traj_dir} --mask_path {MASK_PATH} \\")
    print(f"    --output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
