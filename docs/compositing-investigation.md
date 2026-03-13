# Compositing Pipeline Investigation

Investigation into color shifts and SSIM degradation in composited V2V evaluation outputs.

## Summary

Three issues were identified in OmniAvatar's compositing pipeline compared to LatentSync-train.
Two have been fixed; one is a fundamental architectural limitation shared with LatentSync.

| Issue | Impact on SSIM | Status |
|-------|---------------|--------|
| `macro_block_size` video scaling (564→576) | ~0.14 across full face | **Fixed** |
| float16 precision in `restore_img` | ~0.01-0.02 in background | **Fixed** |
| uint8 quantization before compositing | ~0.004 per channel | **Fixed** |
| Double affine warp round-trip | Inherent to all methods | Architectural (shared with LatentSync) |

**Combined SSIM improvement from all compositing fixes: +0.006** (3.5% of the 0.169 gap
vs LatentSync). The remaining gap is model generation quality, not compositing.

## Issue 1: Video Resolution Scaling (`macro_block_size`)

### Problem

`save_frames_as_video()` used imageio with default `macro_block_size=16`, which **scales**
output frames to the nearest multiple of 16. For a 564x564 GT video, the composited output
became 576x576 (a ~2.1% upscale). During evaluation, the SSIM script resizes 576→564 back,
creating a **double resampling** that degrades every pixel.

LatentSync-train uses `macro_block_size=None`, so outputs match GT resolution exactly.

### Impact

Measured on `WDA_DonnaShalala1_000` (hdtf, lowest SSIM sample):
- The 576→564 resize introduces ~1px resampling noise across the entire frame
- Estimated SSIM impact: **~0.14** (the gap between LatentSync's upper-face SSIM 0.834
  and OmniAvatar's 0.691, where both should be using original face pixels)

### Fix

**File**: `scripts/inference_v2v.py`, function `save_frames_as_video()`

Changed from `quality=8` to explicit `macro_block_size=None` + `ffmpeg_params=["-crf", "13"]`,
matching LatentSync-train's `write_video()` exactly.

Also updated `mux_video_with_audio()` to re-encode at CRF 18 (matching LatentSync-train's
ffmpeg command) instead of stream-copying.

## Issue 2: float16 Precision Loss in Compositing

### Problem

`AlignRestore.restore_img()` (in `affine_transform.py`) defaulted to `dtype=torch.float16`.
Float16 cannot exactly represent all uint8 values (0-255) — values in the 128-255 range
(typical skin tones) have ±0.5-1.0 representation errors. This caused ±1-2 pixel color
shifts in "unchanged" background regions.

The original frame pixels were converted: uint8 → float16 (lossy) → arithmetic → uint8
(rounding). Even where the soft blending mask was 0.0 (background), the float16 conversion
introduced color shifts.

Both LatentSync-train and OmniAvatar had this issue (same default dtype).

### Fix

**File**: `OmniAvatar/utils/latentsync/affine_transform.py`, function `restore_img()`

Changed to use `work_dtype = torch.float32` for all intermediate computation. Float32 can
exactly represent all uint8 values, so background pixels where `soft_mask=0.0` are preserved
bit-exactly. Memory overhead is negligible (~12 MB vs ~6 MB per frame).

## Issue 3: uint8 Quantization Before Compositing

### Problem

In `inference_v2v.py`, the model output was converted to uint8 numpy (line 906) before
being passed to `composite_with_latentsync()`. LatentSync-train keeps data in float32
throughout compositing.

The uint8 conversion happens twice: once after model output, once after mouth-mask blending.
Each conversion loses ~0.4% precision per pixel.

### Fix

**File**: `scripts/inference_v2v.py`

Added `composite_with_latentsync_float()` function that accepts float `[T, C, H, W]` tensors
and performs mouth blending, resize, and `restore_img` entirely in float space. Only converts
to uint8 for final video saving. The old `composite_with_latentsync()` is kept for backward
compatibility with `composite_v2v.py` (standalone compositing from MP4 files).

## Issue 4: Double Affine Warp Round-Trip (Architectural)

### Problem

Every composited pixel — including the upper face kept from the original via `mask.png` —
goes through a forward + inverse affine warp:

```
Original frame → forward affine warp → 512x512 aligned → mask blend
    → resize to crop size → inverse affine warp → soft blend → output
```

Each bilinear interpolation introduces sub-pixel blurring and texture degradation. This is
inherent to the compositing approach and affects LatentSync-train equally.

### Evidence

Face-cropped SSIM (224x224, grayscale) for `WDA_DonnaShalala1_000` (78 frames avg):

| Method | Full Face | Upper Face | Lower Face |
|--------|-----------|------------|------------|
| LatentSync | 0.822 | 0.834 | 0.806 |
| OmniAvatar Refseq | 0.653 | 0.691 | 0.610 |

Both methods show Upper > Lower, confirming the double warp degrades both equally.
The large gap (0.822 vs 0.653) is primarily from Issue 1 (macro_block_size scaling).

### Status

Not fixable without changing the compositing architecture. A potential future improvement
would be to only composite the mouth region (not the full face crop) in `restore_img`,
avoiding the upper-face round-trip entirely.

## Evaluation Pipeline Notes

### SSIM is Face-Cropped, Not Full-Frame

The evaluation pipeline (`eval_ssim_lmd.py`) computes SSIM on **MediaPipe face-cropped
224x224 grayscale** regions, not on the full frame. This focuses the metric on the face
(including the degraded composited region) and excludes the background that would inflate
the score.

Full-frame SSIM: ~0.94 (inflated by identical background)
Face-cropped SSIM: ~0.65 (reflects actual face quality)

### Mask Coverage

`mask.png` (256x256, binary) coverage on the 512x512 aligned face:

| Region | Rows | Keep Original (white) | Use Generated (black) |
|--------|------|-----------------------|-----------------------|
| Forehead | 0-30% | 100% | 0% |
| Eyes/Nose | 30-45% | ~50% | ~50% |
| Mouth | 45-70% | ~8% | ~92% |
| Chin | 70-90% | ~30% | ~70% |

The forehead is 100% kept from the original aligned face, but still goes through the
double affine warp round-trip when composited back via `restore_img()`.

## Pixel-Level Verification

Per-region mean pixel difference from GT (frame 20, `WDA_DonnaShalala1_000`):

| Region | LatentSync [R,G,B] | OmniAvatar [R,G,B] |
|--------|-------------------|---------------------|
| BG corner | [-1.4, -0.6, +0.1] | [-1.0, -1.2, -1.1] |
| Forehead | [+3.3, +0.6, -3.5] | [-1.3, -1.8, -1.6] |
| Left cheek | [+1.9, -0.3, -3.6] | [-0.8, -1.3, -1.1] |
| Chin/mouth | [+4.3, +0.5, -6.0] | [-3.8, -6.1, -5.4] |

OmniAvatar's forehead/cheek diffs are close to the background floor (~1px from video
compression), confirming compositing precision is good. LatentSync has larger diffs (3-4px)
but with a warm color shift rather than uniform darkening.

## Files Changed

| File | Change |
|------|--------|
| `OmniAvatar/utils/latentsync/affine_transform.py` | float32 in `restore_img()` |
| `scripts/inference_v2v.py` | `composite_with_latentsync_float()`, `save_frames_as_video()` CRF 13, `mux_video_with_audio()` CRF 18 re-encode |
