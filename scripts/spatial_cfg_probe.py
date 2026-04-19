"""Spatial CFG-Difference Probe — mechanistic where/when does CFG change predictions.

For each (sample, timestep), feed the teacher the SAME x_t with CFG=4.5 and CFG=1.0.
Decode both predictions via VAE and compute |pred_CFG - pred_noCFG| per-pixel, partitioned
by the LatentSync mask (mouth vs upper_face).

Three protocols (pass --protocol):
  fresh_noise         : x_t = (1-t)·x_0 + t·eps  (fixed per-sample eps across timesteps)
  trajectory_cfg      : x_t loaded from the CFG=4.5 trajectory's saved step_XXX_xt.pt
  trajectory_nocfg    : x_t loaded from the noCFG trajectory's saved step_XXX_xt.pt

Optional noise-floor control (--noise_floor): repeat the two teacher calls at the SAME
CFG but on x_t vs x_t+δ  (δ ~ N(0, σ²), σ small). Measures input-sensitivity baseline.

Outputs per sample:
  spatial_cfg_probe.csv   — scalar stats: step,t,sample,metric,region,value,protocol
  maps/{sample}/step_XXX_diff.npy  — 2D [H,W] diff map averaged over frames
  (optionally) maps/{sample}/step_XXX_nf.npy  — 2D [H,W] noise-floor map

Usage:
    # Protocol 1 (fresh-noise matched)
    CUDA_VISIBLE_DEVICES=2 python scripts/spatial_cfg_probe.py \\
        --protocol fresh_noise \\
        --output_dir /home/work/.local/ode_analysis/spatial_cfg_probe/fresh_noise \\
        --cfg_drop_text false --samples <comma-separated>

    # Protocol 2 (trajectory-CFG matched)
    CUDA_VISIBLE_DEVICES=2 python scripts/spatial_cfg_probe.py \\
        --protocol trajectory_cfg \\
        --traj_dir /home/work/.local/ode_full_trajectories/14B_audio_only_cfg \\
        --output_dir /home/work/.local/ode_analysis/spatial_cfg_probe/trajectory_cfg \\
        --cfg_drop_text false

    # Protocol 3 (trajectory-noCFG matched)
    CUDA_VISIBLE_DEVICES=2 python scripts/spatial_cfg_probe.py \\
        --protocol trajectory_nocfg \\
        --traj_dir /home/work/.local/ode_full_trajectories/14B \\
        --output_dir /home/work/.local/ode_analysis/spatial_cfg_probe/trajectory_nocfg \\
        --cfg_drop_text false
"""

import argparse
import csv
import json
import os
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

# Reuse helpers
from generate_single_step_predictions import (
    load_mask, load_condition, decode_latents,
    PRETRAINED, BASE_14B, CKPT_14B, VAE_PATH, MASK_PATH,
    DATA_DIR, NEG_TEXT_EMB,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=str, required=True,
                   choices=["fresh_noise", "trajectory_cfg", "trajectory_nocfg"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--traj_dir", type=str, default=None,
                   help="Required for trajectory_* protocols. Directory with step_XXX_xt.pt files.")
    p.add_argument("--samples", type=str, default=None,
                   help="Comma-separated sample names. Default: all valid samples.")
    p.add_argument("--cfg_high", type=float, default=4.5, help="CFG value (e.g. 4.5)")
    p.add_argument("--cfg_low",  type=float, default=1.0, help="noCFG value (typically 1.0)")
    p.add_argument("--cfg_drop_text", type=str, default="false", choices=["true", "false"],
                   help="If false, neg branch keeps positive text (audio-only CFG).")
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise_floor", action="store_true",
                   help="Also compute noise-floor baseline (x_t vs x_t+δ at same CFG).")
    p.add_argument("--noise_floor_sigma", type=float, default=0.01,
                   help="Gaussian σ for noise-floor perturbation (in latent units).")
    p.add_argument("--noise_floor_cfg", type=float, default=1.0,
                   help="CFG used when computing noise-floor baseline.")
    p.add_argument("--save_maps", action="store_true", default=True,
                   help="Save 512x512 diff maps as .npy (default on).")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def get_shifted_t_list(num_steps, shift, max_t=0.999):
    t = np.linspace(max_t, 0, num_steps + 1, dtype=np.float64)
    if shift != 1.0:
        t = shift * t / (1 + (shift - 1) * t)
    t[0] = max_t
    t[-1] = 0.0
    return t.astype(np.float32)


def teacher_predict_x0(teacher, x_t, t_val, condition, neg_condition, cfg):
    t_tensor = torch.tensor([t_val], device=x_t.device, dtype=torch.float32)
    x0_cond = teacher(x_t, t_tensor, condition=condition, fwd_pred_type="x0")
    if cfg != 1.0:
        x0_uncond = teacher(x_t, t_tensor, condition=neg_condition, fwd_pred_type="x0")
        return x0_uncond + cfg * (x0_cond - x0_uncond)
    return x0_cond


def compute_region_diff(diff_hw, mask_512):
    """Given a 2D diff map [H,W] and a binary [H,W] mask (1=mouth, 0=upper_face),
    return dict of region means."""
    mouth = mask_512.bool()
    uf = ~mouth
    return {
        "mouth":      float(diff_hw[mouth].mean()),
        "upper_face": float(diff_hw[uf].mean()),
        "full":       float(diff_hw.mean()),
    }


def compute_diff_map(frames_a_np, frames_b_np):
    """
    frames_*: [T, H, W, 3] uint8. Returns:
      diff_map_raw [H,W] float32 — mean(|a-b|) over T and RGB channels (uint8 scale)
      rel_map      [H,W] float32 — mean(|a-b| / (|a|+|b|+eps)) over T and RGB channels
    """
    a = frames_a_np.astype(np.float32)
    b = frames_b_np.astype(np.float32)
    absdiff = np.abs(a - b)                 # [T,H,W,3]
    denom = np.abs(a) + np.abs(b) + 1.0     # +1 uint8 epsilon
    rel = absdiff / denom                    # [T,H,W,3]
    return absdiff.mean(axis=(0, 3)), rel.mean(axis=(0, 3))


def load_trajectory_xt(traj_dir, sample, step_idx, device, dtype):
    path = os.path.join(traj_dir, sample, f"step_{step_idx:03d}_xt.pt")
    if not os.path.exists(path):
        return None
    x = torch.load(path, map_location=device, weights_only=True).to(dtype)
    if x.dim() == 4:
        x = x.unsqueeze(0)  # add batch
    return x


def fresh_noise_xt(gt_latent, eps, t, dtype):
    return ((1.0 - t) * gt_latent + t * eps).to(dtype)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    maps_dir = os.path.join(args.output_dir, "maps")
    if args.save_maps:
        os.makedirs(maps_dir, exist_ok=True)

    if args.protocol in ("trajectory_cfg", "trajectory_nocfg"):
        assert args.traj_dir is not None, f"--traj_dir required for protocol {args.protocol}"

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Samples
    if args.samples:
        samples = args.samples.split(",")
    else:
        samples = sorted([d for d in os.listdir(DATA_DIR)
                          if os.path.isdir(os.path.join(DATA_DIR, d))])

    # Timestep schedule
    t_list = get_shifted_t_list(args.num_steps, args.shift)
    with open(os.path.join(args.output_dir, "schedule.json"), "w") as f:
        json.dump({"t_list": t_list.tolist(), "num_steps": args.num_steps,
                   "shift": args.shift, "protocol": args.protocol,
                   "cfg_high": args.cfg_high, "cfg_low": args.cfg_low,
                   "cfg_drop_text": args.cfg_drop_text,
                   "noise_floor": args.noise_floor,
                   "noise_floor_sigma": args.noise_floor_sigma}, f, indent=2)

    # Masks
    mask_latent = load_mask(64, 64)               # [64,64] bool-ish
    mask_512 = load_mask(512, 512).cpu().numpy()  # [512,512] 0/1

    # Load teacher
    print(f"Loading 14B teacher...")
    teacher = OmniAvatarWan(
        model_size="14B", in_dim=65, mode="v2v", use_audio=True,
        base_model_paths=BASE_14B, omniavatar_ckpt_path=CKPT_14B,
        merge_lora=True, net_pred_type="flow", schedule_type="rf",
    ).to(device, dtype=dtype).eval()
    teacher.requires_grad_(False)

    # Load VAE
    print("Loading Wan VAE...")
    mm = ModelManager(device="cpu", infer=True)
    mm.load_models([VAE_PATH], torch_dtype=torch.bfloat16, device="cpu")
    vae = mm.model[mm.model_name.index("wan_video_vae")].to(device)

    # Negative text embedding
    neg_text_embeds = torch.load(NEG_TEXT_EMB, map_location="cpu", weights_only=False)
    if isinstance(neg_text_embeds, dict):
        neg_text_embeds = next(v for v in neg_text_embeds.values() if isinstance(v, torch.Tensor))
    neg_text_embeds = neg_text_embeds.to(dtype)

    # Open CSV
    csv_path = os.path.join(args.output_dir, "spatial_cfg_probe.csv")
    resume_header = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    f_csv = open(csv_path, "a", newline="")
    writer = csv.writer(f_csv)
    if not resume_header:
        writer.writerow(["step", "t", "sample", "metric", "region", "value", "protocol"])

    overall_start = time.time()
    for si, sample in enumerate(samples):
        sample_start = time.time()
        print(f"\n[{si+1}/{len(samples)}] {sample}")

        # Load sample conditioning
        condition, neg_condition, gt_latents = load_condition(
            sample, mask_latent, neg_text_embeds, device, dtype,
            cfg_drop_text=(args.cfg_drop_text == "true"),
        )
        gt_latents_b = gt_latents.unsqueeze(0).to(device)  # [1, C, F, H, W]

        # Fixed per-sample eps for fresh-noise protocol
        gen = torch.Generator(device=device).manual_seed(args.seed + hash(sample) % (2**31))
        eps = torch.randn(gt_latents_b.shape, generator=gen, device=device, dtype=dtype)

        # Optional per-sample perturbation for noise-floor
        if args.noise_floor:
            nf_eps = torch.randn(gt_latents_b.shape, generator=gen,
                                 device=device, dtype=torch.float32) * args.noise_floor_sigma

        sample_map_dir = os.path.join(maps_dir, sample) if args.save_maps else None
        if sample_map_dir:
            os.makedirs(sample_map_dir, exist_ok=True)

        for step_idx in range(args.num_steps):
            t_val = float(t_list[step_idx])

            marker = os.path.join(sample_map_dir, f"step_{step_idx:03d}_diff.npy") if sample_map_dir else None
            if args.skip_existing and marker and os.path.exists(marker):
                continue

            # Build x_t according to protocol
            if args.protocol == "fresh_noise":
                x_t = fresh_noise_xt(gt_latents_b, eps, t_val, dtype)
            else:
                x_t = load_trajectory_xt(args.traj_dir, sample, step_idx, device, dtype)
                if x_t is None:
                    print(f"  [step {step_idx}] missing xt in traj — skipping")
                    continue

            # Two teacher passes at same x_t
            with torch.no_grad():
                x0_cfg = teacher_predict_x0(teacher, x_t, t_val, condition, neg_condition, args.cfg_high)
                x0_nocfg = teacher_predict_x0(teacher, x_t, t_val, condition, neg_condition, args.cfg_low)

            # Decode both (remove batch dim for VAE)
            with torch.no_grad():
                frames_cfg = decode_latents(vae, x0_cfg.squeeze(0), device)     # [T,H,W,3] uint8
                frames_nocfg = decode_latents(vae, x0_nocfg.squeeze(0), device)

            diff_raw_hw, diff_rel_hw = compute_diff_map(frames_cfg, frames_nocfg)

            r_raw = compute_region_diff(diff_raw_hw, mask_512)
            r_rel = compute_region_diff(diff_rel_hw, mask_512)

            for region, v in r_raw.items():
                writer.writerow([step_idx, f"{t_val:.4f}", sample,
                                 "cfg_diff_raw", region, v, args.protocol])
            for region, v in r_rel.items():
                writer.writerow([step_idx, f"{t_val:.4f}", sample,
                                 "cfg_diff_relative", region, v, args.protocol])

            if args.save_maps and sample_map_dir:
                np.save(os.path.join(sample_map_dir, f"step_{step_idx:03d}_diff.npy"),
                        diff_raw_hw.astype(np.float32))

            # Noise-floor baseline
            if args.noise_floor:
                x_t_perturbed = (x_t.float() + nf_eps).to(dtype)
                with torch.no_grad():
                    x0_a = teacher_predict_x0(teacher, x_t,           t_val, condition, neg_condition, args.noise_floor_cfg)
                    x0_b = teacher_predict_x0(teacher, x_t_perturbed, t_val, condition, neg_condition, args.noise_floor_cfg)
                    fa = decode_latents(vae, x0_a.squeeze(0), device)
                    fb = decode_latents(vae, x0_b.squeeze(0), device)
                nf_raw_hw, _ = compute_diff_map(fa, fb)
                for region, v in compute_region_diff(nf_raw_hw, mask_512).items():
                    writer.writerow([step_idx, f"{t_val:.4f}", sample,
                                     "noise_floor", region, v, args.protocol])
                if args.save_maps and sample_map_dir:
                    np.save(os.path.join(sample_map_dir, f"step_{step_idx:03d}_nf.npy"),
                            nf_raw_hw.astype(np.float32))

            if step_idx % 10 == 0:
                f_csv.flush()

            del x_t, x0_cfg, x0_nocfg, frames_cfg, frames_nocfg

        f_csv.flush()
        print(f"  sample done in {time.time() - sample_start:.0f}s "
              f"(total {time.time() - overall_start:.0f}s)")

    f_csv.close()
    print(f"\nDone. CSV: {csv_path}")
    print(f"Diff maps: {maps_dir}" if args.save_maps else "")


if __name__ == "__main__":
    main()
