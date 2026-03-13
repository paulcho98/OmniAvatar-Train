# Metrics Evaluation Findings

## Discovery: MediaPipe Independent Crop Misalignment

### The Problem

The standard evaluation pipeline (`latentsync-metrics/eval/`) runs MediaPipe face detection
**independently** on GT and generated videos. Each video gets its own bounding box, which means
the 224x224 face crops compared by SSIM/FID/FVD are from **different spatial regions**.

```
GT frame (564x564) → MediaPipe → bbox_gt → crop to 224x224
Gen frame (564x564) → MediaPipe → bbox_gen → crop to 224x224  (DIFFERENT bbox!)
SSIM(gt_crop, gen_crop)  ← comparing misaligned regions
```

### Why the bboxes differ

Even with matching resolutions, the generated mouth changes the face appearance enough to
shift MediaPipe's detected bounding box by 2-5 pixels. Methods with more accurate mouth
generation (e.g., LatentSync) produce smaller bbox shifts (~2px) than our models (~4-5px).

### Quantitative Impact (DonnaShalala, HDTF)

| Crop Method | LatentSync | Refseq_new | Maskall |
|---|---|---|---|
| Standard (independent crops) | 0.826 | 0.649 | — |
| GT-aligned (same bbox) | 0.940 | 0.840 | — |
| Delta | **+0.114** | **+0.191** | — |

The crop misalignment inflates SSIM degradation by 0.11-0.19 points.

### Which metrics are affected?

| Metric | Face detection? | Affected? | Severity |
|---|---|---|---|
| SSIM | MediaPipe, independent | **Yes** | High (~0.1 inflation) |
| FID | MediaPipe, independent | **Yes** | Very high (2-3x inflation) |
| FVD | MediaPipe, independent | **Yes** | High (~30% inflation) |
| LMD | dlib, full frame | No | — |
| CSIM | None (full frame 112x112) | No | — |
| SyncNet | S3FD detector | Separate detector, own behavior |

### Full metrics comparison (HDTF)

| Metric | Eval Method | LatentSync | Maskall-5500 | Refseq_new-3000 |
|---|---|---|---|---|
| **SSIM** | Standard | 0.878 | 0.793 | 0.773 |
| | GT-aligned | 0.967 | 0.912 | 0.902 |
| | Direct 512x512 | — | 0.910 | 0.900 |
| **FID** | Standard | 8.51 | 17.96 | 28.07 |
| | GT-aligned | 4.48 | 5.92 | 13.17 |
| | Direct 512x512 | — | 6.58 | 13.69 |
| **FVD** | Standard | ~155 | 163.8 | 195.6 |
| | GT-aligned | 79.5 | 120.6 | 135.2 |
| | Direct 512x512 | — | 115.8 | 135.1 |

### Why the upper face shows degradation in SSIM heatmaps

The heatmap pattern (red at eyes/edges, green at smooth forehead) is because a small crop
shift (2-5px) affects **high-frequency regions** (eyes, hair boundaries) more than
**smooth regions** (forehead, cheeks). Neighboring pixels in smooth areas look similar, so
a small shift barely changes SSIM. But at texture boundaries (eyelashes, hair-skin edge),
a small shift compares completely different local patterns.

The compositing pipeline replaces the **entire face** (not just mouth) via:
1. Forward affine warp → 512x512
2. Mouth mask blend at 512x512
3. Inverse affine warp → replace face on original frame

So upper face pixels are warped copies (not originals), but the warp round-trip alone
causes only SSIM=0.992 degradation — nearly perfect. The visible upper-face degradation
in heatmaps is almost entirely from crop misalignment.

### The mask boundary

The LatentSync mouth mask transitions at ~50% height (below the nose bridge), NOT at the
eyes. The eye region is 100% kept from the warped original. The forehead (top 49%) has
mask=1.0 (keep original).

## Recommendations

1. **Report both standard and GT-aligned metrics** for fair comparison
2. **GT-aligned eval** should be the primary metric for compositing-based methods
3. **Direct 512x512 eval** (no face detection) is the purest measure of model quality
4. The standard pipeline is still useful for comparing against published baselines
   (since other papers likely use the same independent-crop approach)

## Evaluation Pipeline Details

### `/home/work/.local/latentsync-metrics/eval/`

Same repo at two locations: `latentsync-metrics/` and `eval_metrics/` (minor diffs in
`video_key.py` pattern matching and `run_metrics.sh` defaults).

**Scripts:**
- `eval_fvd.py` — FVD via I3D on frames 20-36, MediaPipe crops to 224x224
- `eval_fid_videos.py` — FID via InceptionV3, MediaPipe crops to 224x224
- `eval_csim.py` — CSIM via ArcFace R100, full frame resized to 112x112 (NO face detection)
- `eval_ssim_lmd.py` — SSIM (MediaPipe 224x224 grayscale) + LMD (dlib 68-point, full frame)
- `eval_sync.py` — SyncNet (S3FD face detector, own pipeline)

**CSV column order:** `step,dataset,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD`

### Our GT-aligned eval script

`scripts/eval_aligned_crops.py` — detects face in GT only, uses same bbox for both.
Computes SSIM, FID (InceptionV3), FVD (I3D). No CSIM/LMD/SyncNet (these don't use
MediaPipe crops and are unaffected).

### Direct 512x512 eval

For pre-compositing 512x512 aligned outputs: resize both GT and generated to 224x224
without any face detection. Purest model quality measure.
