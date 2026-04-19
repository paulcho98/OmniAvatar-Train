# ODE Analysis — LatentSync 1.6 Replication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicate the OmniAvatar ODE trajectory analysis for LatentSync 1.6 at 512×512 on the same 10 validation samples, using the same metrics pipeline, so per-timestep metric curves can be overlaid across models.

**Architecture:** Two new generation scripts under `/home/work/.local/LatentSync/scripts/` mirror the FastGen + OmniAvatar generator pair. Analyze **one canonical chunk of 16 frames per sample** (frames 0–15 of the pre-aligned `sub_clip.mp4`). DDIM noise prediction is converted to x0 via the closed form. A single `--vae_type {wan,sd}` flag on the shared `eval_ode_perceptual_v2.py` selects the correct decoder. Metrics, mask, SyncNet, and dlib all stay identical.

**Tech Stack:** Python, PyTorch, diffusers (DDIM, AutoencoderKL), LatentSync 1.6 checkpoints, existing OmniAvatar eval code. Run in a new conda env `latentsync` (the existing one).

---

## Assumptions and Prerequisites

- `/home/work/stableavatar_data/v2v_validation_data/recon/*/sub_clip.mp4` is **pre-aligned** 512×512, 25fps, 250 frames per sample. We use the first 16 frames directly; no face detection needed.
- LatentSync 1.6 checkpoint at `/home/work/.local/LatentSync/checkpoints/latentsync_unet.pt` (5 GB, Dec 2025).
- Whisper tiny at `/home/work/.local/LatentSync/checkpoints/whisper/tiny.pt`.
- Analysis mask (for metrics, not for LatentSync inference) at `/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png`.
- LatentSync uses its own mouth mask at `/home/work/.local/LatentSync/latentsync/utils/mask.png` during inference — do not conflate the two.

## File Structure

**Create:**
- `/home/work/.local/LatentSync/scripts/generate_latentsync_ode_pairs_full.py` — 50-step DDIM trajectory, saves `step_XXX_xt.pt`, `step_XXX_x0.pt`, `ode_schedule.json`, `input_latents.pt` per sample.
- `/home/work/.local/LatentSync/scripts/generate_latentsync_single_step.py` — Euler-jump / fresh-noise variants.
- `/home/work/.local/OmniAvatar/scripts/plot_all_models_compare.py` — 3-way cross-model overlay.

**Modify:**
- `/home/work/.local/OmniAvatar/scripts/eval_ode_perceptual_v2.py` — add `--vae_type {wan,sd}` flag; Phase 1 (decode) branches on the flag. Phase 2 (metrics) unchanged.

**Outputs produced:**
- `/home/work/.local/ode_full_trajectories/latentsync_1.6/{,_nocfg}/` — trajectory latents (SD-VAE 4-channel).
- `/home/work/.local/ode_analysis/latentsync_1.6/{perceptual_v2,trajectory_nocfg,euler_cfg15_cfg15,euler_nocfg_cfg15,euler_nocfg_nocfg,fresh_noise,combined}/`
- `/home/work/.local/ode_analysis/all_models_combined/` — cross-model overlay.

---

## Task 1: Understand and map LatentSync inference to trajectory format

**Files:** Read-only.

- [ ] **Step 1: Confirm the DDIM step math we need**

From `diffusers.DDIMScheduler`:
- `alphas_cumprod[t]` indexed by integer timestep t (in `[0, 999]`, descending through `scheduler.timesteps`).
- Forward: `x_t = sqrt(α̅_t) · x_0 + sqrt(1-α̅_t) · ε`.
- Reverse: given predicted noise `ε̂`, `x_0 = (x_t - sqrt(1-α̅_t) · ε̂) / sqrt(α̅_t)`.

This is what we use in both generators. No documentation task; just verify before writing code:

```bash
/home/work/.local/miniconda3/envs/latentsync/bin/python -c "
from diffusers import DDIMScheduler
s = DDIMScheduler.from_pretrained('/home/work/.local/LatentSync/configs')
s.set_timesteps(50, device='cpu')
print('timesteps:', s.timesteps[:5].tolist(), '...', s.timesteps[-5:].tolist())
print('alphas_cumprod shape:', s.alphas_cumprod.shape)
print('alpha at t[0]:', s.alphas_cumprod[s.timesteps[0]].item())
print('alpha at t[-1]:', s.alphas_cumprod[s.timesteps[-1]].item())
"
```

Expected: 50 timesteps descending from ~981 to ~0, `alpha_at_t[0]` ≈ 0 (pure noise), `alpha_at_t[-1]` ≈ 1 (clean).

- [ ] **Step 2: Confirm `latentsync` conda env exists**

```bash
/home/work/.local/miniconda3/envs/latentsync/bin/python -c "import diffusers, torch, latentsync; print(diffusers.__version__, torch.__version__)"
```

If the env is missing, fall back to `/home/work/.local/miniconda3/envs/omniavatar/bin/python` and install missing packages; record the env used for downstream tasks.

---

## Task 2: Write `generate_latentsync_ode_pairs_full.py`

**Files:**
- Create: `/home/work/.local/LatentSync/scripts/generate_latentsync_ode_pairs_full.py`

**Context:** Mirrors `/home/work/.local/hyunbin/FastGen/scripts/generate_omniavatar_ode_pairs_full.py` but uses LatentSync's UNet, SD VAE, DDIM scheduler, and whisper tiny audio encoder. Runs one 16-frame chunk per sample.

- [ ] **Step 1: Write the script**

Save the full file below to `/home/work/.local/LatentSync/scripts/generate_latentsync_ode_pairs_full.py`:

```python
"""Generate FULL DDIM trajectory from LatentSync 1.6 — all timesteps + denoised outputs.

For each DDIM step i (0..num_steps-1), saves two files per sample:
  - step_{i:03d}_xt.pt    — noisy latent x_t at timestep t_i  [4, 16, 64, 64]
  - step_{i:03d}_x0.pt    — denoised prediction derived from noise_pred via DDIM

Also saves:
  - ode_schedule.json     — full timestep schedule + metadata
  - input_latents.pt      — ground-truth VAE latent of the first 16 aligned frames [4, 16, 64, 64]

Uses one 16-frame chunk per sample (first 16 frames of sub_clip.mp4).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_latentsync_ode_pairs_full.py \
        --inference_ckpt_path checkpoints/latentsync_unet.pt \
        --unet_config_path configs/unet/stage2_512.yaml \
        --data_dir /home/work/stableavatar_data/v2v_validation_data/recon \
        --output_dir /home/work/.local/ode_full_trajectories/latentsync_1.6 \
        --num_inference_steps 50 --guidance_scale 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask
from latentsync.whisper.audio2feature import Audio2Feature


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inference_ckpt_path", type=str, required=True)
    p.add_argument("--unet_config_path", type=str, default="configs/unet/stage2_512.yaml")
    p.add_argument("--scheduler_config_dir", type=str, default="configs")
    p.add_argument("--whisper_ckpt_path", type=str, default="checkpoints/whisper/tiny.pt")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=1.5)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--video_fps", type=int, default=25)
    p.add_argument("--audio_feat_length", type=int, nargs=2, default=[2, 2])
    p.add_argument("--max_samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def read_first_n_frames(video_path: str, n: int) -> np.ndarray:
    """Read first n frames from video as [n, H, W, 3] uint8 RGB."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)


def vae_encode(vae, images, dtype, device):
    """Encode [F, 3, H, W] in [-1, 1] to [F, 4, H/8, W/8] with LatentSync scaling."""
    images = images.to(device=device, dtype=dtype)
    latents = vae.encode(images).latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents


def load_sample(
    sample_dir: str,
    image_processor: ImageProcessor,
    audio_encoder: Audio2Feature,
    vae: AutoencoderKL,
    num_frames: int,
    video_fps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Dict]:
    video_path = os.path.join(sample_dir, "sub_clip.mp4")
    audio_path = os.path.join(sample_dir, "audio.wav")
    if not (os.path.exists(video_path) and os.path.exists(audio_path)):
        return None

    frames_rgb = read_first_n_frames(video_path, num_frames)
    if len(frames_rgb) < num_frames:
        return None
    frames_tensor = torch.from_numpy(frames_rgb)  # [F, H, W, 3] uint8

    # LatentSync's preprocess expects aligned faces; sub_clip.mp4 is already aligned 512x512.
    pixel_values, masked_pixel_values, masks = image_processor.prepare_masks_and_masked_images(
        frames_tensor, affine_transform=False
    )
    # pixel_values: [F, 3, 512, 512] in [-1, 1]; masked_pixel_values: same; masks: [F, 1, 512, 512]

    # VAE encode reference (full aligned face) and masked face
    with torch.no_grad():
        ref_latents_2d = vae_encode(vae, pixel_values, dtype, device)          # [F, 4, 64, 64]
        masked_image_latents_2d = vae_encode(vae, masked_pixel_values, dtype, device)  # [F, 4, 64, 64]
        gt_latents_2d = ref_latents_2d  # GT for comparison = VAE encoding of pixel_values

    # Resize mask to latent resolution
    mask_latent = torch.nn.functional.interpolate(
        masks.to(device=device, dtype=dtype),
        size=(image_processor.resolution // 8, image_processor.resolution // 8),
    )  # [F, 1, 64, 64]

    # Rearrange to 3D UNet layout [1, C, F, H, W]
    ref_latents = rearrange(ref_latents_2d, "f c h w -> 1 c f h w")
    masked_image_latents = rearrange(masked_image_latents_2d, "f c h w -> 1 c f h w")
    mask_latent = rearrange(mask_latent, "f c h w -> 1 c f h w")
    gt_latents = rearrange(gt_latents_2d, "f c h w -> 1 c f h w")

    # Audio features via whisper
    whisper_feature = audio_encoder.audio2feat(audio_path)
    whisper_chunks = audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
    if len(whisper_chunks) < num_frames:
        return None
    audio_embeds = torch.stack(whisper_chunks[:num_frames]).to(device=device, dtype=dtype)
    # Shape depends on config: for stage2_512 (cross_attention_dim=384, audio_feat_length=[2,2])
    # expect [F, 5, 384] — exact shape comes from feature2chunks.

    return {
        "gt_latents": gt_latents,                        # [1, 4, F, 64, 64]
        "ref_latents": ref_latents,                      # [1, 4, F, 64, 64]
        "masked_image_latents": masked_image_latents,    # [1, 4, F, 64, 64]
        "mask_latent": mask_latent,                      # [1, 1, F, 64, 64]
        "audio_embeds": audio_embeds,                    # [F, ..., 384]
    }


@torch.no_grad()
def extract_full_ddim_trajectory(
    unet: UNet3DConditionModel,
    scheduler: DDIMScheduler,
    sample: Dict,
    num_steps: int,
    guidance_scale: float,
    output_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
):
    ref_latents = sample["ref_latents"]                   # [1, 4, F, 64, 64]
    masked_image_latents = sample["masked_image_latents"] # [1, 4, F, 64, 64]
    mask_latent = sample["mask_latent"]                   # [1, 1, F, 64, 64]
    audio_embeds = sample["audio_embeds"]                 # [F, ..., 384]

    do_cfg = guidance_scale > 1.0

    if do_cfg:
        null_audio = torch.zeros_like(audio_embeds)
        audio_in = torch.cat([null_audio, audio_embeds], dim=0)
        mask_in = torch.cat([mask_latent] * 2, dim=0)
        masked_in = torch.cat([masked_image_latents] * 2, dim=0)
        ref_in = torch.cat([ref_latents] * 2, dim=0)
    else:
        audio_in = audio_embeds
        mask_in = mask_latent
        masked_in = masked_image_latents
        ref_in = ref_latents

    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps  # descending integers

    # Initial noise
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (1, 4, 1, 64, 64)
    noise_single_frame = torch.randn(shape, generator=gen, device=device, dtype=dtype)
    latents = noise_single_frame.repeat(1, 1, ref_latents.shape[2], 1, 1)
    latents = latents * scheduler.init_noise_sigma

    # Save schedule
    schedule_info = {
        "t_list": [int(t.item()) for t in timesteps],
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "scheduler": "DDIM",
        "latent_shape": list(latents.shape[1:]),
    }
    with open(os.path.join(output_dir, "ode_schedule.json"), "w") as f:
        json.dump(schedule_info, f, indent=2)

    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

    for step_idx, t in enumerate(timesteps):
        # Save x_t
        torch.save(
            latents.squeeze(0).to(torch.bfloat16).cpu(),
            os.path.join(output_dir, f"step_{step_idx:03d}_xt.pt"),
        )

        # Build UNet input
        if do_cfg:
            latents_in = torch.cat([latents] * 2, dim=0)
        else:
            latents_in = latents
        latents_in = scheduler.scale_model_input(latents_in, t)
        unet_input = torch.cat([latents_in, mask_in, masked_in, ref_in], dim=1)

        noise_pred = unet(unet_input, t, encoder_hidden_states=audio_in).sample

        if do_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Derive x0_pred via DDIM closed form
        alpha_bar = alphas_cumprod[t]
        sqrt_alpha_bar = alpha_bar.sqrt().to(dtype)
        sqrt_one_minus_alpha_bar = (1 - alpha_bar).sqrt().to(dtype)
        x0_pred = (latents - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar

        torch.save(
            x0_pred.squeeze(0).to(torch.bfloat16).cpu(),
            os.path.join(output_dir, f"step_{step_idx:03d}_x0.pt"),
        )

        # Scheduler step produces next x_t
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return {"num_steps": num_steps}


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.float16 if torch.cuda.get_device_capability()[0] > 7 else torch.float32

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load config
    config = OmegaConf.load(args.unet_config_path)

    # Scheduler
    scheduler = DDIMScheduler.from_pretrained(args.scheduler_config_dir)

    # Audio encoder (whisper tiny for cross_attention_dim=384)
    audio_encoder = Audio2Feature(
        model_path=args.whisper_ckpt_path,
        device="cuda",
        num_frames=args.num_frames,
        audio_feat_length=list(args.audio_feat_length),
    )

    # VAE (SD)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    vae = vae.to(device).eval()

    # UNet
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )
    unet = unet.to(device=device, dtype=dtype).eval()

    # Image processor (512x512, fixed mask from LatentSync)
    mask_image = load_fixed_mask(args.resolution, "/home/work/.local/LatentSync/latentsync/utils/mask.png")
    image_processor = ImageProcessor(args.resolution, device="cuda", mask_image=mask_image)

    # Gather samples
    all_dirs = sorted([
        os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ])
    if args.max_samples:
        all_dirs = all_dirs[:args.max_samples]

    os.makedirs(args.output_dir, exist_ok=True)

    for sample_dir in tqdm(all_dirs, desc="samples"):
        name = os.path.basename(sample_dir)
        out_dir = os.path.join(args.output_dir, name)

        if args.skip_existing and os.path.exists(os.path.join(out_dir, "ode_schedule.json")):
            print(f"Skipping {name}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()

        sample = load_sample(
            sample_dir, image_processor, audio_encoder, vae,
            args.num_frames, args.video_fps, device, dtype,
        )
        if sample is None:
            print(f"Load failed: {name}")
            continue

        # Save GT latents
        torch.save(
            sample["gt_latents"].squeeze(0).to(torch.bfloat16).cpu(),
            os.path.join(out_dir, "input_latents.pt"),
        )

        extract_full_ddim_trajectory(
            unet=unet, scheduler=scheduler, sample=sample,
            num_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            output_dir=out_dir, device=device, dtype=dtype,
            seed=args.seed,
        )

        print(f"  done {name} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script loads without syntax errors**

```bash
cd /home/work/.local/LatentSync
/home/work/.local/miniconda3/envs/latentsync/bin/python -c "
import ast
with open('scripts/generate_latentsync_ode_pairs_full.py') as f:
    ast.parse(f.read())
print('OK')
"
```

---

## Task 3: Smoke test LatentSync trajectory generation (1 sample, 5 steps)

**Files:** No code changes.

- [ ] **Step 1: Run a fast smoke test**

```bash
cd /home/work/.local/LatentSync
mkdir -p /tmp/latentsync_smoke
PY=/home/work/.local/miniconda3/envs/latentsync/bin/python

CUDA_VISIBLE_DEVICES=0 $PY scripts/generate_latentsync_ode_pairs_full.py \
    --inference_ckpt_path checkpoints/latentsync_unet.pt \
    --unet_config_path configs/unet/stage2_512.yaml \
    --data_dir /home/work/stableavatar_data/v2v_validation_data/recon \
    --output_dir /tmp/latentsync_smoke \
    --num_inference_steps 5 --guidance_scale 1.5 --max_samples 1
```

Expected: finishes in <2 min, produces `/tmp/latentsync_smoke/<sample>/{step_000_xt.pt, step_000_x0.pt, ..., step_004_xt.pt, step_004_x0.pt, ode_schedule.json, input_latents.pt}`.

- [ ] **Step 2: Verify shapes**

```bash
PY=/home/work/.local/miniconda3/envs/latentsync/bin/python
$PY -c "
import torch, glob, os
d = glob.glob('/tmp/latentsync_smoke/*/')[0]
xt = torch.load(os.path.join(d, 'step_000_xt.pt'), map_location='cpu', weights_only=True)
x0 = torch.load(os.path.join(d, 'step_000_x0.pt'), map_location='cpu', weights_only=True)
gt = torch.load(os.path.join(d, 'input_latents.pt'), map_location='cpu', weights_only=True)
print('xt:', xt.shape, xt.dtype, 'range', xt.float().min().item(), xt.float().max().item())
print('x0:', x0.shape, x0.dtype, 'range', x0.float().min().item(), x0.float().max().item())
print('gt:', gt.shape, gt.dtype, 'range', gt.float().min().item(), gt.float().max().item())
"
```

Expected: all shapes `[4, 16, 64, 64]`, dtype `torch.bfloat16`. x0 at early step should be close to zero (very noisy estimate), gt should be in the usual SD VAE latent range (~ ±5).

- [ ] **Step 3: Smoke test — decode `input_latents.pt` and compare to raw frames**

This validates that the VAE encode/decode round-trip is consistent.

```bash
PY=/home/work/.local/miniconda3/envs/latentsync/bin/python
$PY -c "
import sys, torch, glob, os, cv2, numpy as np
sys.path.insert(0, '/home/work/.local/LatentSync')
from diffusers import AutoencoderKL
from einops import rearrange

d = glob.glob('/tmp/latentsync_smoke/*/')[0]
sample_name = os.path.basename(d.rstrip('/'))
gt = torch.load(os.path.join(d, 'input_latents.pt'), map_location='cuda', weights_only=True).float()
print('gt shape:', gt.shape)

vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse', torch_dtype=torch.float32).to('cuda').eval()
vae.config.scaling_factor = 0.18215
vae.config.shift_factor = 0

# Decode: undo LatentSync scaling
latents_flat = rearrange(gt.unsqueeze(0), 'b c f h w -> (b f) c h w')
latents_flat = latents_flat / vae.config.scaling_factor + vae.config.shift_factor
with torch.no_grad():
    decoded = vae.decode(latents_flat).sample  # [F, 3, 512, 512] in [-1, 1]
decoded_rgb = ((decoded.clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

# Read original frames
cap = cv2.VideoCapture(f'/home/work/stableavatar_data/v2v_validation_data/recon/{sample_name}/sub_clip.mp4')
orig = []
for i in range(16):
    ok, f = cap.read()
    orig.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
cap.release()
orig = np.stack(orig)  # [16, 512, 512, 3]

mse = ((orig.astype(np.float32) - decoded_rgb.astype(np.float32))**2).mean()
print('orig mean:', orig.mean(), 'decoded mean:', decoded_rgb.mean())
print('VAE round-trip pixel MSE (uint8 scale):', mse)
print('Relative MSE (fraction of 255^2):', mse / (255**2))
"
```

Expected: round-trip pixel MSE in uint8 scale < 100 (compared to a random pair ~10000). Relative MSE < 0.002. If much higher, the SD VAE scaling is wrong.

- [ ] **Step 4: Commit the generator**

```bash
cd /home/work/.local/LatentSync
git add scripts/generate_latentsync_ode_pairs_full.py
git commit -m "feat: LatentSync 1.6 DDIM trajectory extractor for ODE analysis"
```

---

## Task 4: Full LatentSync trajectory — CFG=1.5 on 10 samples, 50 steps

**Files:** No code changes.

- [ ] **Step 1: Run generator on 10 samples**

```bash
cd /home/work/.local/LatentSync
PY=/home/work/.local/miniconda3/envs/latentsync/bin/python
OUT=/home/work/.local/ode_full_trajectories/latentsync_1.6
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 nohup $PY scripts/generate_latentsync_ode_pairs_full.py \
    --inference_ckpt_path checkpoints/latentsync_unet.pt \
    --unet_config_path configs/unet/stage2_512.yaml \
    --data_dir /home/work/stableavatar_data/v2v_validation_data/recon \
    --output_dir "$OUT" \
    --num_inference_steps 50 --guidance_scale 1.5 \
    --max_samples 10 --skip_existing \
    > "$OUT/trajectory.log" 2>&1 &

echo "PID=$!"
```

Expected time: ~20–30 min (UNet is much smaller than Wan 14B).

- [ ] **Step 2: Verify completion**

```bash
tail -n 40 /home/work/.local/ode_full_trajectories/latentsync_1.6/trajectory.log
ls /home/work/.local/ode_full_trajectories/latentsync_1.6/ | wc -l
```

Expected: 10 directories, each with 102 files.

---

## Task 5: LatentSync noCFG trajectory (CFG=1.0, only cond branch)

**Files:** No code changes.

- [ ] **Step 1: Run with guidance_scale=1.0**

```bash
cd /home/work/.local/LatentSync
PY=/home/work/.local/miniconda3/envs/latentsync/bin/python
OUT=/home/work/.local/ode_full_trajectories/latentsync_1.6_nocfg
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=1 nohup $PY scripts/generate_latentsync_ode_pairs_full.py \
    --inference_ckpt_path checkpoints/latentsync_unet.pt \
    --unet_config_path configs/unet/stage2_512.yaml \
    --data_dir /home/work/stableavatar_data/v2v_validation_data/recon \
    --output_dir "$OUT" \
    --num_inference_steps 50 --guidance_scale 1.0 \
    --max_samples 10 --skip_existing \
    > "$OUT/trajectory.log" 2>&1 &

echo "PID=$!"
```

Can run in parallel with Task 4 on a different GPU.

---

## Task 6: Add `--vae_type` flag to `eval_ode_perceptual_v2.py`

**Files:**
- Modify: `/home/work/.local/OmniAvatar/scripts/eval_ode_perceptual_v2.py`

**Context:** Phase 1 (decode) currently hardcodes Wan VAE via `from OmniAvatar.models.model_manager import ModelManager`. Add a branch for SD VAE. Phase 2 and merge are unchanged.

- [ ] **Step 1: Add the CLI flag**

Find the argument parser (search for `--phase`). Add:

```python
    parser.add_argument("--vae_type", type=str, default="wan",
                        choices=["wan", "sd"],
                        help="VAE family for decoding. 'wan' = OmniAvatar Wan VAE (16ch), "
                             "'sd' = Stable Diffusion VAE via diffusers (4ch).")
```

- [ ] **Step 2: Modify `decode_latents_to_numpy` to handle both**

Find the current implementation:

```python
def decode_latents_to_numpy(vae, latents, device):
    """Decode [16, 21, 64, 64] → [T, H, W, 3] uint8 numpy."""
    latents = latents.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        frames = vae.decode(latents, device=device, tiled=False)
    if frames.dim() == 5:
        frames = frames[0]
    frames = ((frames.float() + 1) / 2 * 255).clamp(0, 255).byte()
    return frames.permute(1, 2, 3, 0).cpu().numpy()
```

Replace with a dispatcher:

```python
def decode_latents_to_numpy(vae, latents, device, vae_type="wan"):
    """Decode latents to [T, H, W, 3] uint8 numpy, VAE-type agnostic."""
    if vae_type == "wan":
        # Wan VAE: expects [1, 16, F, H, W], returns [1, 3, F, H, W] in [-1, 1]
        latents = latents.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            frames = vae.decode(latents, device=device, tiled=False)
        if frames.dim() == 5:
            frames = frames[0]
        frames = ((frames.float() + 1) / 2 * 255).clamp(0, 255).byte()
        return frames.permute(1, 2, 3, 0).cpu().numpy()

    elif vae_type == "sd":
        # SD VAE (diffusers AutoencoderKL): latents saved as [4, F, H, W] at LatentSync scaling.
        # Undo scaling, decode 2D per frame.
        from einops import rearrange
        sf = vae.config.scaling_factor
        shift = vae.config.shift_factor
        latents_5d = latents.unsqueeze(0)  # [1, 4, F, H, W]
        latents_2d = rearrange(latents_5d, "b c f h w -> (b f) c h w")
        latents_2d = latents_2d.to(device=device, dtype=next(vae.parameters()).dtype)
        latents_2d = latents_2d / sf + shift
        with torch.no_grad():
            dec = vae.decode(latents_2d).sample  # [F, 3, H, W] in [-1, 1]
        dec = ((dec.clamp(-1, 1) + 1) / 2 * 255).byte()  # uint8
        return dec.permute(0, 2, 3, 1).cpu().numpy()

    else:
        raise ValueError(f"Unknown vae_type: {vae_type}")
```

- [ ] **Step 3: Modify `run_decode` to load either VAE**

Find `run_decode(args)`. The current implementation hardcodes:

```python
    from OmniAvatar.models.model_manager import ModelManager
    ...
    model_manager = ModelManager(...)
    model_manager.load_models([args.vae_path], ...)
    vae = model_manager.model[vae_idx].to(device)
```

Change to:

```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_names = shard_list(discover_samples(args.traj_dir), args.shard_id, args.num_shards)
    t_list, num_steps = load_schedule(args.traj_dir, discover_samples(args.traj_dir))

    if args.shard_id is not None:
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(sample_names)} samples")

    if args.vae_type == "wan":
        from OmniAvatar.models.model_manager import ModelManager
        print("Loading Wan VAE...")
        model_manager = ModelManager(device="cpu", infer=True)
        model_manager.load_models([args.vae_path], torch_dtype=torch.bfloat16, device="cpu")
        vae_idx = model_manager.model_name.index("wan_video_vae")
        vae = model_manager.model[vae_idx].to(device)
    elif args.vae_type == "sd":
        from diffusers import AutoencoderKL
        print("Loading SD VAE (stabilityai/sd-vae-ft-mse)...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        vae = vae.to(device).eval()
    else:
        raise ValueError(f"Unknown vae_type: {args.vae_type}")
```

Also find the calls to `decode_latents_to_numpy(...)` inside `run_decode` and pass `vae_type=args.vae_type`.

- [ ] **Step 4: Handle Wan-VAE-only `--vae_path` requirement gracefully**

If `args.vae_type == "sd"`, `--vae_path` may be absent. Make it optional:

```python
    parser.add_argument("--vae_path", type=str, default=None,
                        help="Path to Wan VAE .pth (required when --vae_type wan)")
```

Add an assertion early in `run_decode`:

```python
    if args.vae_type == "wan" and args.vae_path is None:
        raise ValueError("--vae_path is required when --vae_type wan")
```

- [ ] **Step 5: Regression smoke test — decode Wan trajectory still works**

Run the existing Wan decode on ONE sample (reduce `--num_shards` to simulate):

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
rm -rf /tmp/wan_decode_regression
$PY scripts/eval_ode_perceptual_v2.py --phase decode \
    --vae_type wan \
    --traj_dir /home/work/.local/ode_full_trajectories/14B \
    --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
    --output_dir /tmp/wan_decode_regression \
    --shard_id 0 --num_shards 10  # processes 1 sample
```

Expected: completes without error, produces videos.

- [ ] **Step 6: Decode the LatentSync smoke-test output**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
$PY scripts/eval_ode_perceptual_v2.py --phase decode \
    --vae_type sd \
    --traj_dir /tmp/latentsync_smoke \
    --output_dir /tmp/latentsync_decode_smoke
```

Expected: produces `/tmp/latentsync_decode_smoke/videos/<sample>/step_004.mp4` (the final step) visually similar to the GT lip motion.

- [ ] **Step 7: Commit**

```bash
cd /home/work/.local/OmniAvatar
git add scripts/eval_ode_perceptual_v2.py
git commit -m "feat: --vae_type {wan,sd} flag in eval_ode_perceptual_v2"
```

---

## Task 7: Full decode + metrics on LatentSync CFG=1.5 trajectory

**Files:** No code changes.

- [ ] **Step 1: Phase 1 — decode (4 GPUs)**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
TRAJ=/home/work/.local/ode_full_trajectories/latentsync_1.6
OUT=/home/work/.local/ode_analysis/latentsync_1.6/perceptual_v2
mkdir -p "$OUT"

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD $PY scripts/eval_ode_perceptual_v2.py \
        --phase decode --vae_type sd \
        --traj_dir "$TRAJ" --output_dir "$OUT" \
        --shard_id $SHARD --num_shards 4 > "$OUT/decode_shard${SHARD}.log" 2>&1 &
done
wait
```

Expected time: ~20 min (4-channel SD VAE is fast, 16 frames only).

- [ ] **Step 2: Phase 2 — metrics**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
TRAJ=/home/work/.local/ode_full_trajectories/latentsync_1.6
OUT=/home/work/.local/ode_analysis/latentsync_1.6/perceptual_v2
MASK=/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD $PY scripts/eval_ode_perceptual_v2.py \
        --phase metrics \
        --traj_dir "$TRAJ" --mask_path "$MASK" --output_dir "$OUT" \
        --shard_id $SHARD --num_shards 4 > "$OUT/metrics_shard${SHARD}.log" 2>&1 &
done
wait
```

- [ ] **Step 3: Merge + plot**

```bash
cd /home/work/.local/OmniAvatar
$PY scripts/eval_ode_perceptual_v2.py --merge \
    --traj_dir "$TRAJ" --output_dir "$OUT"
```

- [ ] **Step 4: Repeat for noCFG trajectory**

```bash
TRAJ2=/home/work/.local/ode_full_trajectories/latentsync_1.6_nocfg
OUT2=/home/work/.local/ode_analysis/latentsync_1.6/trajectory_nocfg
mkdir -p "$OUT2"

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD $PY scripts/eval_ode_perceptual_v2.py \
        --phase decode --vae_type sd \
        --traj_dir "$TRAJ2" --output_dir "$OUT2" \
        --shard_id $SHARD --num_shards 4 > "$OUT2/decode_shard${SHARD}.log" 2>&1 &
done
wait

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD $PY scripts/eval_ode_perceptual_v2.py \
        --phase metrics \
        --traj_dir "$TRAJ2" --mask_path "$MASK" --output_dir "$OUT2" \
        --shard_id $SHARD --num_shards 4 > "$OUT2/metrics_shard${SHARD}.log" 2>&1 &
done
wait

$PY scripts/eval_ode_perceptual_v2.py --merge \
    --traj_dir "$TRAJ2" --output_dir "$OUT2"
```

---

## Task 8: Write `generate_latentsync_single_step.py`

**Files:**
- Create: `/home/work/.local/LatentSync/scripts/generate_latentsync_single_step.py`

**Context:** Mirrors `generate_single_step_predictions.py` for OmniAvatar. Two modes: `euler_jump` (load saved `step_k_xt.pt`, one UNet pass with CFG `cfg_step0`, derive x0) and `fresh_noise` (sample fresh eps at each step, compute x_t = sqrt(α̅)·x_0 + sqrt(1-α̅)·eps, one UNet pass).

- [ ] **Step 1: Write the script**

Save to `/home/work/.local/LatentSync/scripts/generate_latentsync_single_step.py`:

```python
"""Single-step predictions from LatentSync 1.6 at each DDIM timestep.

Modes:
  fresh_noise: at each step k, x_t = sqrt(α̅_t)·x_0 + sqrt(1-α̅_t)·ε (fixed-seed ε)
  euler_jump:  at each step k, load saved step_k_xt.pt from trajectory, run UNet once

Both modes produce the same artifact layout as generate_latentsync_ode_pairs_full.py:
  step_{i:03d}_x0.pt, ode_schedule.json
(We don't re-save xt since it's loaded from the parent trajectory.)

Usage:
    python scripts/generate_latentsync_single_step.py \
        --mode euler_jump \
        --traj_dir /home/work/.local/ode_full_trajectories/latentsync_1.6 \
        --output_dir /home/work/.local/ode_analysis/latentsync_1.6/euler_cfg15_cfg15 \
        --inference_ckpt_path checkpoints/latentsync_unet.pt \
        --cfg_step0 1.5 --skip_existing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.utils.image_processor import ImageProcessor, load_fixed_mask
from latentsync.whisper.audio2feature import Audio2Feature

# Reuse the helpers from the trajectory script
sys.path.insert(0, os.path.dirname(__file__))
from generate_latentsync_ode_pairs_full import load_sample  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, choices=["euler_jump", "fresh_noise"])
    p.add_argument("--traj_dir", type=str, required=True,
                   help="Parent trajectory dir (for input_latents.pt and step_XXX_xt.pt)")
    p.add_argument("--data_dir", type=str,
                   default="/home/work/stableavatar_data/v2v_validation_data/recon")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--inference_ckpt_path", type=str, required=True)
    p.add_argument("--unet_config_path", type=str,
                   default="/home/work/.local/LatentSync/configs/unet/stage2_512.yaml")
    p.add_argument("--scheduler_config_dir", type=str,
                   default="/home/work/.local/LatentSync/configs")
    p.add_argument("--whisper_ckpt_path", type=str,
                   default="/home/work/.local/LatentSync/checkpoints/whisper/tiny.pt")
    p.add_argument("--samples", type=str, default=None,
                   help="Comma-separated sample names; default = all in traj_dir")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=1.5,
                   help="Used only for fresh_noise mode")
    p.add_argument("--cfg_step0", type=float, default=1.5,
                   help="CFG scale for the UNet forward pass (euler_jump mode)")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--video_fps", type=int, default=25)
    p.add_argument("--audio_feat_length", type=int, nargs=2, default=[2, 2])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


@torch.no_grad()
def run_unet_single_step(unet, scheduler, latents, sample, t, guidance_scale, dtype):
    audio = sample["audio_embeds"]
    mask_ = sample["mask_latent"]
    masked = sample["masked_image_latents"]
    ref = sample["ref_latents"]

    do_cfg = guidance_scale > 1.0
    if do_cfg:
        audio_in = torch.cat([torch.zeros_like(audio), audio], dim=0)
        mask_in = torch.cat([mask_] * 2, dim=0)
        masked_in = torch.cat([masked] * 2, dim=0)
        ref_in = torch.cat([ref] * 2, dim=0)
        latents_in = torch.cat([latents] * 2, dim=0)
    else:
        audio_in = audio
        mask_in = mask_
        masked_in = masked
        ref_in = ref
        latents_in = latents

    latents_in = scheduler.scale_model_input(latents_in, t)
    unet_input = torch.cat([latents_in, mask_in, masked_in, ref_in], dim=1)
    noise_pred = unet(unet_input, t, encoder_hidden_states=audio_in).sample

    if do_cfg:
        u, c = noise_pred.chunk(2)
        noise_pred = u + guidance_scale * (c - u)

    return noise_pred


def derive_x0(x_t, noise_pred, alpha_bar, dtype):
    sqrt_alpha_bar = alpha_bar.sqrt().to(dtype)
    sqrt_one_minus = (1 - alpha_bar).sqrt().to(dtype)
    return (x_t - sqrt_one_minus * noise_pred) / sqrt_alpha_bar


@torch.no_grad()
def run_sample(sample_dir_out, sample_name, mode, traj_dir, args,
               unet, scheduler, image_processor, audio_encoder, vae,
               device, dtype):

    data_sample_dir = os.path.join(args.data_dir, sample_name)
    loaded = load_sample(
        data_sample_dir, image_processor, audio_encoder, vae,
        args.num_frames, args.video_fps, device, dtype,
    )
    if loaded is None:
        print(f"Load failed: {sample_name}")
        return

    traj_sample_dir = os.path.join(traj_dir, sample_name)
    gt_latents = torch.load(os.path.join(traj_sample_dir, "input_latents.pt"),
                            map_location=device, weights_only=True).to(dtype).unsqueeze(0)  # [1,4,F,H,W]

    scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

    # Save schedule (same as trajectory's)
    schedule_info = {
        "t_list": [int(t.item()) for t in timesteps],
        "num_steps": args.num_inference_steps,
        "guidance_scale": args.cfg_step0 if mode == "euler_jump" else args.guidance_scale,
        "scheduler": "DDIM",
        "mode": mode,
    }
    with open(os.path.join(sample_dir_out, "ode_schedule.json"), "w") as f:
        json.dump(schedule_info, f, indent=2)

    gen = torch.Generator(device=device).manual_seed(args.seed)

    for step_idx, t in enumerate(timesteps):
        out_path = os.path.join(sample_dir_out, f"step_{step_idx:03d}_x0.pt")
        if args.skip_existing and os.path.exists(out_path):
            continue

        alpha_bar = alphas_cumprod[t]

        if mode == "euler_jump":
            x_t_path = os.path.join(traj_sample_dir, f"step_{step_idx:03d}_xt.pt")
            x_t = torch.load(x_t_path, map_location=device, weights_only=True).to(dtype).unsqueeze(0)
            cfg = args.cfg_step0
        else:  # fresh_noise
            eps = torch.randn(gt_latents.shape, generator=gen, device=device, dtype=dtype)
            sqrt_a = alpha_bar.sqrt().to(dtype)
            sqrt_1ma = (1 - alpha_bar).sqrt().to(dtype)
            x_t = sqrt_a * gt_latents + sqrt_1ma * eps
            cfg = args.guidance_scale

        noise_pred = run_unet_single_step(unet, scheduler, x_t, loaded, t, cfg, dtype)
        x0_pred = derive_x0(x_t, noise_pred, alpha_bar, dtype)

        torch.save(x0_pred.squeeze(0).to(torch.bfloat16).cpu(), out_path)

    # Also save input_latents for eval downstream
    torch.save(gt_latents.squeeze(0).to(torch.bfloat16).cpu(),
               os.path.join(sample_dir_out, "input_latents.pt"))


def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.float16 if torch.cuda.get_device_capability()[0] > 7 else torch.float32

    config = OmegaConf.load(args.unet_config_path)
    scheduler = DDIMScheduler.from_pretrained(args.scheduler_config_dir)

    audio_encoder = Audio2Feature(
        model_path=args.whisper_ckpt_path, device="cuda",
        num_frames=args.num_frames, audio_feat_length=list(args.audio_feat_length),
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype).to(device).eval()
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path, device="cpu",
    )
    unet = unet.to(device=device, dtype=dtype).eval()

    mask_image = load_fixed_mask(args.resolution,
                                  "/home/work/.local/LatentSync/latentsync/utils/mask.png")
    image_processor = ImageProcessor(args.resolution, device="cuda", mask_image=mask_image)

    os.makedirs(args.output_dir, exist_ok=True)

    # Samples: explicit list or all in traj_dir
    if args.samples:
        sample_names = args.samples.split(",")
    else:
        sample_names = sorted([d for d in os.listdir(args.traj_dir)
                                if os.path.isdir(os.path.join(args.traj_dir, d))
                                and os.path.exists(os.path.join(args.traj_dir, d, "ode_schedule.json"))])

    for name in tqdm(sample_names, desc="samples"):
        out = os.path.join(args.output_dir, name)
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        run_sample(out, name, args.mode, args.traj_dir, args,
                   unet, scheduler, image_processor, audio_encoder, vae, device, dtype)
        print(f"  done {name} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

```bash
/home/work/.local/miniconda3/envs/latentsync/bin/python -c "
import ast
with open('/home/work/.local/LatentSync/scripts/generate_latentsync_single_step.py') as f:
    ast.parse(f.read())
print('OK')
"
```

- [ ] **Step 3: Smoke test — euler_jump on 1 sample, 5 steps**

```bash
cd /home/work/.local/LatentSync
PY=/home/work/.local/miniconda3/envs/latentsync/bin/python
CUDA_VISIBLE_DEVICES=0 $PY scripts/generate_latentsync_single_step.py \
    --mode euler_jump \
    --traj_dir /tmp/latentsync_smoke \
    --output_dir /tmp/latentsync_smoke_euler \
    --inference_ckpt_path checkpoints/latentsync_unet.pt \
    --num_inference_steps 5 --cfg_step0 1.5
```

Expected: produces `step_000_x0.pt` through `step_004_x0.pt`. x0 at step 4 should be close to the GT since t is nearly 0.

```bash
$PY -c "
import torch, os, glob
d = glob.glob('/tmp/latentsync_smoke_euler/*/')[0]
gt = torch.load(os.path.join(d, 'input_latents.pt'), map_location='cpu', weights_only=True).float()
for i in range(5):
    x0 = torch.load(os.path.join(d, f'step_{i:03d}_x0.pt'), map_location='cpu', weights_only=True).float()
    err = (x0 - gt).pow(2).mean().sqrt().item()
    print(f'step {i}: x0 RMSE vs GT = {err:.4f}')
"
```

Expected: error decreases monotonically, step 4 error is small (< 1.0 in latent units). Very early steps can have large error — that's expected.

- [ ] **Step 4: Commit**

```bash
cd /home/work/.local/LatentSync
git add scripts/generate_latentsync_single_step.py
git commit -m "feat: LatentSync single-step predictions for ODE analysis"
```

---

## Task 9: Run all 4 single-step variants for LatentSync

**Files:** No code changes.

**GPU:** 4 GPUs, ~1 hour total.

- [ ] **Step 1: Launch 4 variants in parallel**

```bash
cd /home/work/.local/LatentSync
PY=/home/work/.local/miniconda3/envs/latentsync/bin/python
SCRIPT=scripts/generate_latentsync_single_step.py
TRAJ=/home/work/.local/ode_full_trajectories/latentsync_1.6
BASE=/home/work/.local/ode_analysis/latentsync_1.6
CKPT=checkpoints/latentsync_unet.pt
mkdir -p "$BASE"

CUDA_VISIBLE_DEVICES=0 nohup $PY $SCRIPT \
    --mode fresh_noise --traj_dir "$TRAJ" \
    --output_dir "$BASE/fresh_noise" \
    --inference_ckpt_path "$CKPT" \
    --guidance_scale 1.5 --skip_existing \
    > "$BASE/fresh_noise.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup $PY $SCRIPT \
    --mode euler_jump --traj_dir "$TRAJ" \
    --output_dir "$BASE/euler_cfg15_cfg15" \
    --inference_ckpt_path "$CKPT" \
    --cfg_step0 1.5 --skip_existing \
    > "$BASE/euler_cfg15_cfg15.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup $PY $SCRIPT \
    --mode euler_jump --traj_dir "$TRAJ" \
    --output_dir "$BASE/euler_nocfg_cfg15" \
    --inference_ckpt_path "$CKPT" \
    --cfg_step0 1.0 --skip_existing \
    > "$BASE/euler_nocfg_cfg15.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup $PY $SCRIPT \
    --mode euler_jump --traj_dir "$TRAJ" \
    --output_dir "$BASE/euler_nocfg_nocfg" \
    --inference_ckpt_path "$CKPT" \
    --cfg_step0 1.0 --skip_existing \
    > "$BASE/euler_nocfg_nocfg.log" 2>&1 &

wait
```

Note: the "noCFG→CFG" variant label is preserved for plot consistency. For LatentSync, `cfg_step0` is the only CFG knob (unlike OmniAvatar where CFG is applied again in the teacher). The `euler_nocfg_cfg15` variant actually means "step 0 at noCFG" but the label distinguishes it from `euler_nocfg_nocfg` in the plots. Since LatentSync's single-step prediction is identical to its trajectory-level Euler step, `cfg_teacher` is redundant here — document this explicitly in the variant README.

- [ ] **Step 2: Verify each variant produced files**

```bash
for V in fresh_noise euler_cfg15_cfg15 euler_nocfg_cfg15 euler_nocfg_nocfg; do
    N=$(find /home/work/.local/ode_analysis/latentsync_1.6/$V -name "step_*_x0.pt" 2>/dev/null | wc -l)
    echo "$V: $N files (expected 500)"
done
```

---

## Task 10: Decode + metrics on the 4 single-step variants

**Files:** No code changes.

**GPU:** 4 GPUs, ~4–6 hours (SyncNet dominates).

- [ ] **Step 1: Decode all 4 variants**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
TRAJ=/home/work/.local/ode_full_trajectories/latentsync_1.6
BASE=/home/work/.local/ode_analysis/latentsync_1.6

GPU=0
for V in fresh_noise euler_cfg15_cfg15 euler_nocfg_cfg15 euler_nocfg_nocfg; do
    CUDA_VISIBLE_DEVICES=$GPU nohup $PY scripts/eval_ode_perceptual_v2.py \
        --phase decode --vae_type sd \
        --traj_dir "$BASE/$V" --output_dir "$BASE/$V" \
        > "$BASE/$V/decode.log" 2>&1 &
    GPU=$(( (GPU + 1) % 4 ))
done
wait
```

Note: `--traj_dir` points at the variant's own dir (it contains its own `ode_schedule.json` and `input_latents.pt`), not the parent trajectory.

- [ ] **Step 2: Metrics all 4 variants**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
MASK=/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png
BASE=/home/work/.local/ode_analysis/latentsync_1.6

GPU=0
for V in fresh_noise euler_cfg15_cfg15 euler_nocfg_cfg15 euler_nocfg_nocfg; do
    CUDA_VISIBLE_DEVICES=$GPU nohup $PY scripts/eval_ode_perceptual_v2.py \
        --phase metrics --mask_path "$MASK" \
        --traj_dir "$BASE/$V" --output_dir "$BASE/$V" \
        > "$BASE/$V/metrics.log" 2>&1 &
    GPU=$(( (GPU + 1) % 4 ))
done
wait

for V in fresh_noise euler_cfg15_cfg15 euler_nocfg_cfg15 euler_nocfg_nocfg; do
    $PY scripts/eval_ode_perceptual_v2.py --merge \
        --traj_dir "$BASE/$V" --output_dir "$BASE/$V"
done
```

---

## Task 11: LatentSync combined plots

**Files:**
- Create: `/home/work/.local/OmniAvatar/scripts/plot_combined_ode_comparison_latentsync.py`

- [ ] **Step 1: Duplicate the combined plot script with LatentSync paths**

```bash
cp /home/work/.local/OmniAvatar/scripts/plot_combined_ode_comparison.py \
   /home/work/.local/OmniAvatar/scripts/plot_combined_ode_comparison_latentsync.py
```

Open the new file and replace the `VARIANTS` dict at the top:

```python
VARIANTS = {
    "trajectory": {
        "path": "/home/work/.local/ode_analysis/latentsync_1.6/perceptual_v2/metrics.csv",
        "label": "50-step trajectory CFG=1.5",
        "color": "black",
        "linestyle": "-",
        "linewidth": 2.5,
    },
    "trajectory_nocfg": {
        "path": "/home/work/.local/ode_analysis/latentsync_1.6/trajectory_nocfg/metrics.csv",
        "label": "50-step trajectory noCFG",
        "color": "black",
        "linestyle": "--",
        "linewidth": 2.5,
    },
    "euler_cfg15_cfg15": {
        "path": "/home/work/.local/ode_analysis/latentsync_1.6/euler_cfg15_cfg15/metrics.csv",
        "label": "Euler: CFG",
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 1.5,
    },
    "euler_nocfg_cfg15": {
        "path": "/home/work/.local/ode_analysis/latentsync_1.6/euler_nocfg_cfg15/metrics.csv",
        "label": "Euler: noCFG step0",
        "color": "tab:orange",
        "linestyle": "-",
        "linewidth": 1.5,
    },
    "euler_nocfg_nocfg": {
        "path": "/home/work/.local/ode_analysis/latentsync_1.6/euler_nocfg_nocfg/metrics.csv",
        "label": "Euler: noCFG",
        "color": "tab:red",
        "linestyle": "--",
        "linewidth": 1.5,
    },
}

TRAJ_DIR = "/home/work/.local/ode_full_trajectories/latentsync_1.6"
```

Also change the figure suptitles to reference LatentSync.

- [ ] **Step 2: Copy CSVs and generate plots**

```bash
BASE=/home/work/.local/ode_analysis/latentsync_1.6
COMBINED=$BASE/combined
mkdir -p "$COMBINED"

cp $BASE/perceptual_v2/metrics.csv       $COMBINED/trajectory_cfg15.csv
cp $BASE/trajectory_nocfg/metrics.csv    $COMBINED/trajectory_nocfg.csv
cp $BASE/euler_cfg15_cfg15/metrics.csv   $COMBINED/euler_cfg15_cfg15.csv
cp $BASE/euler_nocfg_cfg15/metrics.csv   $COMBINED/euler_nocfg_cfg15.csv
cp $BASE/euler_nocfg_nocfg/metrics.csv   $COMBINED/euler_nocfg_nocfg.csv
cp $BASE/fresh_noise/metrics.csv         $COMBINED/fresh_noise.csv

cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
$PY scripts/plot_combined_ode_comparison_latentsync.py --output_dir "$COMBINED"
```

Expected: `reference_metrics_combined.png` and `noref_metrics_combined.png` under the combined directory.

- [ ] **Step 3: Commit**

```bash
cd /home/work/.local/OmniAvatar
git add scripts/plot_combined_ode_comparison_latentsync.py
git commit -m "feat: combined plots for LatentSync 1.6 ODE analysis"
```

---

## Task 12: Cross-model overlay plot (all 3 models)

**Files:**
- Create: `/home/work/.local/OmniAvatar/scripts/plot_all_models_compare.py`

- [ ] **Step 1: Write the cross-model plot**

Save to `/home/work/.local/OmniAvatar/scripts/plot_all_models_compare.py`:

```python
"""Cross-model overlay: OmniAvatar original CFG vs audio-only CFG vs LatentSync 1.6.

Only the 50-step trajectory curves are overlaid (CFG and noCFG variants per model).
Step index on x-axis. Each model's own scheduler t is labelled on a twin axis.

Usage:
    python scripts/plot_all_models_compare.py --output_dir /home/work/.local/ode_analysis/all_models_combined
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = {
    "omni_original": {
        "csv_cfg":   "/home/work/.local/ode_analysis/14B/perceptual_v2/metrics.csv",
        "csv_nocfg": "/home/work/.local/ode_analysis/14B/trajectory_nocfg/metrics.csv",
        "label_cfg":   "OmniAvatar CFG=4.5 (text+audio)",
        "label_nocfg": "OmniAvatar noCFG",
        "color": "tab:blue",
    },
    "omni_audio_only": {
        "csv_cfg":   "/home/work/.local/ode_analysis/14B_audio_only_cfg/perceptual_v2/metrics.csv",
        "csv_nocfg": "/home/work/.local/ode_analysis/14B/trajectory_nocfg/metrics.csv",
        "label_cfg":   "OmniAvatar CFG=4.5 (audio-only)",
        "label_nocfg": None,  # same as original, avoid duplicate legend
        "color": "tab:orange",
    },
    "latentsync": {
        "csv_cfg":   "/home/work/.local/ode_analysis/latentsync_1.6/perceptual_v2/metrics.csv",
        "csv_nocfg": "/home/work/.local/ode_analysis/latentsync_1.6/trajectory_nocfg/metrics.csv",
        "label_cfg":   "LatentSync 1.6 CFG=1.5",
        "label_nocfg": "LatentSync 1.6 noCFG",
        "color": "tab:green",
    },
}


def load_metrics(csv_path):
    df = pd.read_csv(csv_path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df[df["step"] >= 0]
    return df.groupby(["step", "metric", "region"])["value"].mean().reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    loaded = {}
    for key, m in MODELS.items():
        loaded[key] = {
            "cfg":   load_metrics(m["csv_cfg"])   if os.path.exists(m["csv_cfg"])   else None,
            "nocfg": load_metrics(m["csv_nocfg"]) if os.path.exists(m["csv_nocfg"]) else None,
        }
        print(f"{key}: cfg={loaded[key]['cfg'] is not None}, nocfg={loaded[key]['nocfg'] is not None}")

    ref_metrics = [
        ("pixel_mse", "Pixel MSE (mouth)", True),
        ("ssim",      "SSIM (mouth)",      False),
        ("lpips",     "LPIPS (mouth)",     False),
        ("lmd",       "LMD (mouth)",       False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    fig.suptitle("Cross-Model 50-Step Trajectory Comparison (mouth region)",
                 fontsize=16, fontweight="bold")

    for col, (metric, title, use_log) in enumerate(ref_metrics):
        ax = axes[col]
        for key, m in MODELS.items():
            for variant, ls, lw, label_key in [
                ("cfg",   "-",  2.2, "label_cfg"),
                ("nocfg", "--", 1.8, "label_nocfg"),
            ]:
                df = loaded[key][variant]
                if df is None or m[label_key] is None:
                    continue
                sub = df[(df["metric"] == metric) & (df["region"] == "mouth")].sort_values("step")
                if len(sub) == 0:
                    continue
                ax.plot(sub["step"], sub["value"],
                        color=m["color"], linestyle=ls, linewidth=lw,
                        label=m[label_key], marker=".", markersize=2)

        ax.set_xlabel("ODE Step Index (0 = most noisy → 49 = clean)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

    plt.tight_layout()
    out = os.path.join(args.output_dir, "all_models_reference.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # No-reference metrics
    noref_metrics = [
        ("sharpness", "Mouth Sharpness (Laplacian var)"),
        ("sync_d",    "Sync-D (lower=better)"),
        ("sync_c",    "Sync-C (higher=better)"),
    ]

    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))
    fig2.suptitle("Cross-Model No-Reference Metrics", fontsize=14, fontweight="bold")

    for col, (metric, title) in enumerate(noref_metrics):
        ax = axes2[col]
        for key, m in MODELS.items():
            for variant, ls, lw, label_key in [
                ("cfg",   "-",  2.2, "label_cfg"),
                ("nocfg", "--", 1.8, "label_nocfg"),
            ]:
                df = loaded[key][variant]
                if df is None or m[label_key] is None:
                    continue
                sub = df[(df["metric"] == metric) & (df["region"] == "mouth")].sort_values("step")
                if len(sub) == 0:
                    continue
                ax.plot(sub["step"], sub["value"],
                        color=m["color"], linestyle=ls, linewidth=lw,
                        label=m[label_key], marker=".", markersize=2)

        ax.set_xlabel("ODE Step Index")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(args.output_dir, "all_models_noref.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the plot**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
mkdir -p /home/work/.local/ode_analysis/all_models_combined
$PY scripts/plot_all_models_compare.py \
    --output_dir /home/work/.local/ode_analysis/all_models_combined
```

Expected: `all_models_reference.png` and `all_models_noref.png`.

- [ ] **Step 3: Commit**

```bash
cd /home/work/.local/OmniAvatar
git add scripts/plot_all_models_compare.py
git commit -m "feat: cross-model ODE trajectory overlay plot"
```

---

## Task 13: Write README for LatentSync combined directory

**Files:**
- Create: `/home/work/.local/ode_analysis/latentsync_1.6/combined/README.md`

- [ ] **Step 1: Write README**

```markdown
# LatentSync 1.6 — ODE Trajectory Analysis

Replicates `/home/work/.local/ode_analysis/14B/combined/` for LatentSync 1.6
at 512×512. Same 10 validation samples, same metrics, same mask.

## Model and inference

- **Model**: LatentSync 1.6 (512×512), `checkpoints/latentsync_unet.pt` (5 GB).
- **UNet**: 3D UNet with 13-channel input (4 noisy latent + 1 mask + 4 masked_image + 4 ref).
- **VAE**: `stabilityai/sd-vae-ft-mse` (SD VAE, 4 latent channels, scaling_factor=0.18215).
- **Audio encoder**: Whisper tiny (cross_attention_dim=384).
- **Scheduler**: DDIM, 50 steps.
- **CFG**: audio-only (LatentSync's native scheme; the unconditional branch zeroes audio while
  keeping mask/masked_image/ref identical). Default CFG=1.5.
- **Chunk**: first 16 frames of each `sub_clip.mp4` (pre-aligned 512×512).

## Variants

| CSV | What |
|---|---|
| `trajectory_cfg15.csv` | Full 50-step DDIM, CFG=1.5, from `generate_latentsync_ode_pairs_full.py` |
| `trajectory_nocfg.csv` | Full 50-step DDIM, CFG=1.0 (conditional only) |
| `euler_cfg15_cfg15.csv` | Euler-jump: at each step, load saved x_t, one UNet pass at CFG=1.5 |
| `euler_nocfg_cfg15.csv` | Euler-jump: at each step, load saved x_t, one UNet pass at CFG=1.0 |
| `euler_nocfg_nocfg.csv` | Same as above (CFG knob is single in LatentSync; kept for layout parity) |
| `fresh_noise.csv` | At each step, x_t from fixed-seed noise + GT latent, one UNet pass at CFG=1.5 |

Note: Unlike OmniAvatar, LatentSync's single-step prediction has only one CFG knob
(`cfg_step0`). The `euler_nocfg_cfg15` and `euler_nocfg_nocfg` variants
are preserved for plot-layout parity with the OmniAvatar comparison.

## Metrics

Identical to `/home/work/.local/ode_analysis/14B/combined/README.md`:
pixel MSE, SSIM, LPIPS, LMD, sharpness (Laplacian var), SyncNet Sync-C/D.
Mouth/upper-face split via `Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png`.

## Caveats for cross-model comparison

- LatentSync analyses 16 frames per sample (~0.64s); OmniAvatar analyses 81 frames (~3.24s).
  Per-timestep averages are thus noisier for LatentSync. Trends should still be robust.
- DDIM timesteps are integer [0, 999] descending; OmniAvatar's flow-matching t is float [0, 1] descending.
  The primary x-axis is **step index (0..49)**; the per-model t value is shown on a twin axis.
- SD VAE (4-channel) vs Wan VAE (16-channel, 3D) — absolute MSE in latent space is not comparable,
  but pixel-space metrics after decoding are.
```

- [ ] **Step 2: Commit**

```bash
cd /home/work/.local/OmniAvatar
git add ../ode_analysis/latentsync_1.6/combined/README.md 2>/dev/null || true
# README lives outside OmniAvatar repo; commit via whatever repo tracks /home/work/.local/ode_analysis/ if any
```

If `/home/work/.local/ode_analysis/` is not tracked by git, skip the commit — the file exists as an artefact.

---

## Final verification

- [ ] All artefacts exist:

```bash
echo "=== LatentSync per-variant metrics ==="
for V in perceptual_v2 trajectory_nocfg euler_cfg15_cfg15 euler_nocfg_cfg15 euler_nocfg_nocfg fresh_noise; do
    F=/home/work/.local/ode_analysis/latentsync_1.6/$V/metrics.csv
    [ -f "$F" ] && echo "  OK $V ($(wc -l < $F) rows)" || echo "  MISSING $V"
done

echo "=== Combined plots ==="
ls /home/work/.local/ode_analysis/latentsync_1.6/combined/*.png 2>/dev/null

echo "=== Cross-model plots ==="
ls /home/work/.local/ode_analysis/all_models_combined/*.png 2>/dev/null
```

Expected: all 6 CSVs present, at least 2 combined plots, 2 cross-model plots.
