# Compositing Evaluation Fix

## Problem

Our composited V2V evaluation videos had a resolution mismatch bug that inflated the
SSIM gap vs LatentSync by causing **MediaPipe crop misalignment** during evaluation.

### Root Cause: `macro_block_size=16` in video saving

`save_frames_as_video()` used imageio with default `macro_block_size=16`, which scales
output frames to the nearest multiple of 16:

| Original resolution | Composited resolution | Scaling |
|---|---|---|
| 510x510 | 512x512 | +0.4% |
| 564x564 | 576x576 | +2.1% |
| 790x790 | 800x800 | +1.3% |
| 1034x1034 | 1040x1040 | +0.6% |

The evaluation pipeline (`eval_ssim_lmd.py`) runs MediaPipe face detection **independently**
on each video. MediaPipe returns **normalized** bounding box coordinates (0.0–1.0), which
are multiplied by the image dimensions. When the composited video is at a different resolution
than GT, the same normalized face position maps to different pixel coordinates:

```
GT video (564x564):         face bbox = [85, 142, 432, 498]
Composited (576x576):       face bbox = [87, 148, 444, 514]
                            Shift:       +2   +6  +12  +16 pixels
```

This creates a **12-16 pixel systematic crop shift** between the two 224x224 face crops
being compared by SSIM. The evaluation is comparing misaligned face regions.

### Impact: +0.18 SSIM degradation from crop misalignment alone

Measured on `WDA_DonnaShalala1_000` (hdtf, lowest SSIM sample):

| Crop method | LatentSync | Refseq | Gap |
|---|---|---|---|
| Independent crops (eval pipeline) | 0.823 | 0.656 | 0.167 |
| Same GT bbox (no misalignment) | 0.940 | 0.840 | 0.100 |

The crop misalignment accounts for +0.184 SSIM degradation on Refseq and +0.117 on
LatentSync (which also has slight shifts of ~1px). The gap narrows from 0.167 to 0.100.

### Secondary issues (minor impact)

- **float16 in `restore_img()`**: ±1-2px color shifts. Fixed → float32.
- **uint8 quantization before compositing**: Minor precision loss. Fixed → float pipeline.

### What is NOT the problem

- **Double affine warp round-trip**: Tested by running GT through forward warp → inverse
  warp with no model generation. Result: SSIM = 0.992. The warp barely degrades anything.
- **Mask boundary at eyes**: The LatentSync mouth mask transitions at ~50% height (below
  nose), well below the eyes. The eyes are 100% kept from the original aligned face.
- **Upper face degradation in heatmaps**: Almost entirely from crop misalignment, not from
  actual pixel degradation. When using the same bbox, upper face SSIM is near-perfect.

## Fix Applied

### Code changes (already in codebase)

1. `save_frames_as_video()` in `scripts/inference_v2v.py`:
   - `macro_block_size=None` + `ffmpeg_params=["-crf", "13"]`
2. `restore_img()` in `OmniAvatar/utils/latentsync/affine_transform.py`:
   - `work_dtype = torch.float32` for all intermediate computation
3. `composite_with_latentsync_float()` in `scripts/inference_v2v.py`:
   - Float-space mouth blending (no uint8 quantization)

### Re-compositing (this fix)

All existing composited evaluation videos are re-composited with the fixed pipeline:
- Old composited dirs archived as `{ds}_composited_old/`
- New composited dirs match original resolution exactly (no scaling)

| Experiment | Steps re-composited |
|---|---|
| maskall | 3000, 4000, 4500, 5500 |
| refseq | 1500 |
| refseq_new | 3000 |

## Expected outcome

- SSIM numbers should improve as crop misalignment is eliminated
- The gap vs LatentSync should narrow (both now at correct resolution)
- Fairer comparison for all metrics that use face cropping (SSIM, FID, CSIM)
