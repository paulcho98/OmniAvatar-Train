# V2V Data Preprocessing

## Overview

`scripts/preprocess_v2v_integrated.py` creates the full precomputed training
data from raw source videos. One script replaces the original multi-step
pipeline (separate VAE, audio, ref-latent, and mask-patching scripts).

### Output per sample directory

```
{video_stem}/
    sub_clip.mp4              # symlink to source video
    audio.wav                 # 16 kHz mono PCM (extracted via ffmpeg)
    prompt.txt                # text prompt
    text_emb.pt               # symlink to pre-computed T5 embedding
    vae_latents_mask_all.pt   # {input_latents, masked_latents} [16, 21, 64, 64] bf16
    audio_emb_omniavatar.pt   # {audio_emb: [num_frames, 10752]} bf16
    ref_latents.pt            # {ref_sequence_latents: [16, 21, 64, 64]} bf16
```

### Prerequisites

- Wan 2.1 VAE checkpoint (`Wan2.1_VAE.pth`)
- wav2vec2-base-960h model directory
- LatentSync mask PNG (`mask.png`)
- Pre-computed T5 text embedding (`.pt` file for `--common_prompt_path`)
- Metadata CSV with columns: `video` (filename), `prompt` (text)

## Usage

### Fresh training run (recommended for new experiments)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/preprocess_v2v_integrated.py \
    --csv_path /path/to/metadata.csv \
    --video_dir /path/to/raw_videos/ \
    --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
    --wav2vec_path pretrained_models/wav2vec2-base-960h \
    --latentsync_mask_path /path/to/mask.png \
    --common_prompt_path /path/to/text_emb.pt \
    --output_dir /path/to/output/ \
    --batch_size 4
```

This masks all frames (including frame 0) in pixel space, then encodes the
entire video through the VAE in a single pass. Produces clean, self-consistent
latents.

### Reproducing existing training data (`--legacy_twostep_mask`)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/preprocess_v2v_integrated.py \
    --legacy_twostep_mask \
    --csv_path /path/to/metadata.csv \
    --video_dir /path/to/raw_videos/ \
    --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
    --wav2vec_path pretrained_models/wav2vec2-base-960h \
    --latentsync_mask_path /path/to/mask.png \
    --common_prompt_path /path/to/text_emb.pt \
    --output_dir /path/to/output/ \
    --batch_size 4
```

**Use this flag when you need latents identical to the existing training
data** (e.g., fine-tuning from an existing checkpoint or reproducing
results).

#### Why the flag exists

The original training data was created by two separate scripts:

1. `precompute_vae_latents_nomask.py` — encodes the full video with frame 0
   **unmasked** in the masked variant. The VAE's causal feature cache from
   the unmasked frame 0 propagates into frames 1+.

2. `precompute_vae_latents_masked.py` — re-encodes **only frame 0** with
   masking applied (single-frame VAE encode) and splices it into position 0.

This two-step process creates a subtle inconsistency: frames 1+ have VAE
causal-conv features influenced by the unmasked frame 0, while frame 0
itself is encoded from masked pixels. The effect is measurable (~14%
relative MAE on frame 1, decaying to <1% by frame 5) but all existing
models were trained on this data, so they expect it.

The default (without `--legacy_twostep_mask`) masks all frames before a
single VAE encode pass. This is more correct (consistent feature cache)
and recommended for training from scratch.

## Other flags

| Flag | Description |
|------|-------------|
| `--force` | Re-process directories even if all output files exist |
| `--skip_gpu` | Phase 1 only (directory setup, no GPU encoding) |
| `--also_save_vae_latents` | Save the legacy `vae_latents.pt` alongside `vae_latents_mask_all.pt` |
| `--batch_size N` | Videos per GPU batch for VAE encoding (default 4) |
| `--num_loader_threads N` | I/O prefetch threads (default 4) |
| `--num_cpu_workers N` | CPU workers for Phase 1 directory setup (default 16) |

## Data size estimates (29K samples)

| Component | Per sample | Total |
|-----------|-----------|-------|
| Raw videos (H.264) | ~430 KB | ~12 GB |
| audio.wav (PCM 16kHz) | ~230 KB | ~6 GB |
| **Transfer total** | | **~18 GB** |
| vae_latents_mask_all.pt | 5.3 MB | ~150 GB |
| audio_emb_omniavatar.pt | 7.7 MB | ~220 GB |
| ref_latents.pt | 2.7 MB | ~77 GB |
| text_emb.pt | 8.1 MB | ~13 GB (deduplicated) |
| **Precomputed total** | ~24 MB | **~669 GB** |
