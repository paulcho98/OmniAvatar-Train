# Session 2026-03-03: 1.3B Training, VAE Tiling Fix, Mask-All-Frames

## Overview

Three features implemented in this session:

1. **1.3B V2V training support** — launch script and config for the smaller model
2. **VAE tiling disabled for 512x512** — fixes color drift at tile boundaries in inference
3. **`--mask_all_frames` + `--no_first_frame_overwrite`** — two independent flags for controlling frame 0 treatment

---

## 1. 1.3B V2V Training Support

### What
Created a launch script for training the 1.3B OmniAvatar model in V2V mode. The codebase was already fully dynamic (all dimensions computed from `num_layers`, model detected by state dict hash), so no code changes were needed — only config/paths.

### Files
- **`scripts/train_v2v_1.3B.sh`** (new) — 4 GPUs, grad_accum=2, effective batch=8
- **`configs/accelerate_4gpus.yaml`** (new) — `MULTI_GPU` + `bf16` + 4 processes
- **`scripts/train_v2v_auxloss_4gpus.sh`** (new) — 14B variant also on 4 GPUs

### Key differences from 14B
| Setting | 14B | 1.3B |
|---------|-----|------|
| DiT weights | 6 sharded safetensors | Single `diffusion_pytorch_model.safetensors` |
| Model paths | `Wan2.1-T2V-14B` / `OmniAvatar-14B` | `Wan2.1-T2V-1.3B` / `OmniAvatar-1.3B` |
| Output path | `output_omniavatar_v2v_auxloss` | `output_omniavatar_v2v_1.3B` |
| Wandb run | `v2v_14B_auxloss` | `v2v_1.3B_auxloss` |

Everything else (learning rate, aux losses, validation, data paths) is identical. The `--dit_paths` arg accepts a single path (no comma) which works because the code does `dit_paths.split(",")`.

### Why no code changes needed
- Audio projection count: `num_layers // 2 - 1` (automatic: 19 for 14B, 14 for 1.3B)
- Model detection: state dict hash in `wan_video_dit.py` auto-selects config
- `in_dim=49` and `audio_hidden_size=32` are model-size-independent
- LoRA config read from checkpoint `config.json`

### Batching investigation
We investigated true batch_size>1 support. Finding: **the entire training codebase is hardcoded for batch_size=1** — collate_fn returns `batch[0]`, timestep sampling uses `(1,)`, scheduler expects scalar timesteps, all data handling uses `unsqueeze(0)`. Gradient accumulation is the mechanism for larger effective batch size. True batching would require ~12 code locations across 3 files.

---

## 2. VAE Tiling Disabled for 512x512

### Problem
Color drift in the masked area during V2V inference. The mouth region's colors would subtly shift compared to the unmasked upper face.

### Root cause
`inference_v2v.py` used tiled VAE encode/decode (`tiled=True`) with hardcoded tile parameters `(34,34)/(18,16)`. At 512x512 (64x64 latent), tiling creates overlapping decode regions with boundary blending, causing color inconsistencies — especially visible at the mask boundary between generated mouth and preserved upper face.

The original I2V `inference.py` also uses tiling (different params: `(30,52)/(15,26)` via `log_video` defaults), but at 512x512 tiling is unnecessary since the full 64x64 latent fits in a single pass.

### Fix
Three lines in `scripts/inference_v2v.py`:
```python
# Encode (2 calls):
self.pipe.encode_video(..., tiled=False)

# Decode (1 call):
self.pipe.decode_video(latents, tiled=False)
```

### Comprehensive inference.py vs inference_v2v.py diff
We did a full comparison and documented all differences. Key unexpected findings beyond tiling:
- **No TeaCache in V2V** — intentional (thresh=0 in config anyway)
- **No separate audio_cfg_scale** — irrelevant (audio_scale=None → same as guidance_scale → 2-pass CFG in both)
- **Single-pass vs chunked** — V2V generates all 81 frames in one pass (doesn't call `pipe.log_video`)
- **V2V implements denoising loop directly** — iterates `scheduler.timesteps` and calls `pipe.dit()` manually

Full analysis saved in plan file: `/home/work/.claude_hb/plans/quizzical-tumbling-map.md`

---

## 3. `--mask_all_frames` + `--no_first_frame_overwrite`

### Background: How frame 0 works
In V2V, frame 0 has special treatment in three places:

| Component | Default (frame 0) | Default (frames 1+) |
|-----------|-------------------|---------------------|
| **Mask channel** (in y tensor) | `0` (keep all) | Spatial mask (0=keep upper, 1=gen mouth) |
| **Masked video** (in y tensor) | Untouched (full face) | Mouth zeroed to 0.0 in [-1,1] |
| **Fixed-frame overwrite** (denoising) | YES — overwritten every step | No |

### What the flags do

**`--mask_all_frames`** (training + inference):
- Mask channel: frame 0 gets the same spatial mask as frames 1+ (not all-zero)
- Masked video: frame 0's mouth is zeroed like other frames (live path)
- Precomputed: loads `vae_latents_mask_all.pt` instead of `vae_latents.pt`

**`--no_first_frame_overwrite`** (validation/inference only):
- Disables `latents[:,:,:1] = lat[:,:,:1]` at each denoising step
- Frame 0 is generated freely instead of being force-anchored to the reference

These are **independent** — any combination of the two works. This enables testing 4 configurations.

### Files modified

**`scripts/train_v2v.py`:**
- `OmniAvatarV2VDataset.__init__()`: added `mask_all_frames` param
- `_getitem_precomputed()`: loads `vae_latents_mask_all.pt` when `mask_all_frames=True`
- `_apply_spatial_mask()`: conditionally masks frame 0 (live path)
- `OmniAvatarV2VTrainingModule.__init__()`: added both params
- `prepare_v2v_input()`: mask channel uses spatial mask for all frames when enabled
- `run_v2v_validation()`: `fixed_frame = 0 if no_first_frame_overwrite else 1`
- `train_parser()`: two new `add_argument` calls
- `main()`: wires both flags to dataset and model constructors

**`scripts/inference_v2v.py`:**
- `apply_spatial_mask_normalized()`: added `mask_all_frames` param
- `prepare_v2v_y()`: added `mask_all_frames` param
- `forward()`: reads both flags from `self.args`, passes through

### Files created

**`scripts/precompute_vae_latents_masked.py`** (new):
- Reads existing `vae_latents.pt`, loads frame 0 from source video
- Applies LatentSync mask to frame 0, VAE-encodes, replaces in `masked_latents`
- Saves as `vae_latents_mask_all.pt` (original file untouched)
- Supports sharding for multi-GPU parallel processing

**`scripts/run_precompute_vae_masked.sh`** (new):
- 4-GPU launcher for the precompute script
- Uses 1.3B VAE weights

### Launch scripts
- `scripts/train_v2v_1.3B.sh`: both flags **active**
- `scripts/train_v2v_auxloss_4gpus.sh`: both flags **commented out**

### Precomputed data compatibility
- `vae_latents.pt` (original): frame 0 unmasked — used when `mask_all_frames=False`
- `vae_latents_mask_all.pt` (new): frame 0 masked — used when `mask_all_frames=True`
- Fallback: if `vae_latents_mask_all.pt` doesn't exist, falls back to `vae_latents.pt`
- Must run precompute script before training with `--mask_all_frames` for correct behavior
