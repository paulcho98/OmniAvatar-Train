# V2V Data Preprocessing

## Quick Start: New Machine Setup

This guide covers transferring raw data and reproducing the full
precomputed training dataset from scratch, targeting the Self-Forcing
Re-DMD training script:

```
FastGen-redmd/scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_t769_fsdpfix.sh
```

### Step 0: Transfer the raw data bundle

On the source machine, package the raw data:

```bash
# Videos (all in one flat directory, ~12 GB)
tar cf videos.tar -C /home/work/.local/combined_data high_visual_quality/

# Audio (scattered across 29K sample dirs, ~6 GB → ~3 GB compressed)
cd /home/work/stableavatar_data/v2v_training_data
find . -name "audio.wav" -print0 | tar czf audio.tar.gz --null -T -

# Metadata CSV + data lists + mask (tiny)
cp metadata_combined_with_voxceleb2.csv /tmp/bundle/
cp video_square_path.txt video_square_path_combined.txt video_square_val10.txt /tmp/bundle/
cp /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png /tmp/bundle/
```

VoxCeleb2 videos (if using the combined 35K list):
```bash
tar cf voxceleb2_videos.tar -C /home/work/data/VoxCeleb2_processed_all high_visual_quality/
# ~8.6 GB
```

**Total transfer: ~22 GB** (or ~30 GB with VoxCeleb2).

### Step 1: Unpack on the target machine

```bash
# Choose a data root
export DATA_ROOT=/path/to/data

# Videos
mkdir -p $DATA_ROOT/raw_videos
tar xf videos.tar -C $DATA_ROOT/raw_videos/

# Audio + metadata
mkdir -p $DATA_ROOT/v2v_training_data
tar xzf audio.tar.gz -C $DATA_ROOT/v2v_training_data/
cp metadata_combined_with_voxceleb2.csv $DATA_ROOT/v2v_training_data/
cp video_square_path*.txt video_square_val*.txt $DATA_ROOT/v2v_training_data/
cp mask.png $DATA_ROOT/
```

### Step 2: Prerequisites (pretrained models)

These must be present on the target machine:

| Model | Used by | Approximate size |
|-------|---------|-----------------|
| `Wan2.1_VAE.pth` | VAE encode (latents + ref) | ~500 MB |
| `wav2vec2-base-960h/` | Audio embedding | ~360 MB |
| `models_t5_umt5-xxl-enc-bf16.pth` | Text embedding | ~20 GB |
| `diffusion_pytorch_model.safetensors` (1.3B) | Student base weights | ~5 GB |
| `diffusion_pytorch_model-*.safetensors` (14B, 6 shards) | Teacher base weights | ~28 GB |
| LatentSync `mask.png` | Spatial masking | tiny |

OmniAvatar-trained checkpoints (also need to transfer):

| Checkpoint | Role | Size |
|------------|------|------|
| 1.3B syncnet-trained V2V adapter (`step-1000.pt`) | Student + fake score init | ~340 MB |
| 14B mouthweight teacher (`step-6000.pt`) | Teacher | ~1.2 GB |
| DF-trained student (`0005000.pth`) | Student init for SF | ~8 GB |
| TAEW decoder (`taew2_1.pth`) | Reward decoder | ~45 MB |
| SyncNet-v2 (`syncnet_v2.model`) | Reward scorer | ~25 MB |

### Step 3: Precompute text embeddings

Text embeddings are per-video T5 encodings of descriptive captions
(8,750 unique prompts across 36K samples). The CSV contains the actual
prompt text, so we can re-encode them:

```bash
cd /path/to/OmniAvatar-Train

python scripts/precompute_text_embeddings.py \
    --csv_path $DATA_ROOT/v2v_training_data/metadata_combined_with_voxceleb2.csv \
    --text_encoder_path pretrained_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth \
    --output_dir $DATA_ROOT/text_emb/ \
    --sample_dir $DATA_ROOT/v2v_training_data/ \
    --device cuda
```

This encodes ~8,750 unique prompts (a few minutes on one GPU), saves each
as `<md5>.pt` in `output_dir`, and creates `text_emb.pt` symlinks in each
sample directory.

Multi-GPU sharding (optional):
```bash
for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$shard python scripts/precompute_text_embeddings.py \
        --csv_path ... --text_encoder_path ... --output_dir ... --sample_dir ... \
        --shard_id $shard --num_shards 4 --device cuda:0 &
done
wait
# Then rebuild all symlinks:
python scripts/precompute_text_embeddings.py \
    --csv_path ... --output_dir ... --sample_dir ... --symlink_only
```

**Note:** Embeddings are saved in fp32 `[1, 512, 4096]`. These will not
be bit-identical to the original hallo3_data embeddings if the T5 model
version or hardware differs, but OmniAvatar training uses 10% text dropout
so small encoding differences are negligible.

### Step 4: Precompute VAE latents, audio embeddings, and reference latents

Use `--legacy_twostep_mask` for consistency with all existing checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/preprocess_v2v_integrated.py \
    --legacy_twostep_mask \
    --csv_path $DATA_ROOT/v2v_training_data/metadata_combined_with_voxceleb2.csv \
    --video_dir $DATA_ROOT/raw_videos/high_visual_quality/ \
    --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
    --wav2vec_path pretrained_models/wav2vec2-base-960h \
    --latentsync_mask_path $DATA_ROOT/mask.png \
    --output_dir $DATA_ROOT/v2v_training_data/ \
    --batch_size 4
```

This produces per-sample: `vae_latents_mask_all.pt`, `audio_emb_omniavatar.pt`,
`ref_latents.pt`.

Estimated time: ~4-8 hours on 4 GPUs for 35K samples.

### Step 5: Update data list paths

The `video_square_path.txt` files contain absolute paths from the source
machine. Update them to point to the new location:

```bash
cd $DATA_ROOT/v2v_training_data
sed -i "s|/home/work/stableavatar_data/v2v_training_data|$DATA_ROOT/v2v_training_data|g" \
    video_square_path.txt video_square_path_combined.txt video_square_val10.txt
```

### Step 6: Set environment variables and launch training

```bash
export OMNIAVATAR_ROOT=/path/to/OmniAvatar-Train
export OMNIAVATAR_DATA_ROOT=$DATA_ROOT/v2v_training_data
export OMNIAVATAR_DATA_LIST=$DATA_ROOT/v2v_training_data/video_square_path.txt
export OMNIAVATAR_VAL_LIST=$DATA_ROOT/v2v_training_data/video_square_val10.txt
export LATENTSYNC_MASK_PATH=$DATA_ROOT/mask.png

# Checkpoint paths (update to match your transfer locations)
export OMNIAVATAR_DF_CKPT=/path/to/df_checkpoints/0005000.pth
export OMNIAVATAR_TEACHER_CKPT=/path/to/teacher/step-6000.pt
export OMNIAVATAR_STUDENT_CKPT=/path/to/student/step-1000.pt

cd /path/to/FastGen-redmd
bash scripts/train_sf_sink1_window7_redmd_beta2_audiofix_taew_syncnet_mouthweight_fsmatched_t769_fsdpfix.sh
```

### Verify data integrity

Quick check that precomputed data loads correctly:

```bash
python -c "
from fastgen.datasets.omniavatar_dataloader import OmniAvatarDataset
ds = OmniAvatarDataset(
    data_list_path='$DATA_ROOT/v2v_training_data/video_square_val10.txt',
    latentsync_mask_path='$DATA_ROOT/mask.png',
    use_ref_sequence=True,
)
sample = ds[0]
for k, v in sample.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: {v.shape} {v.dtype}')
    else:
        print(f'  {k}: {type(v).__name__}')
print(f'OK — {len(ds)} samples loadable')
"
```

---

## Reference: Preprocessing Scripts

### `preprocess_v2v_integrated.py`

Creates VAE latents, audio embeddings, and reference latents from raw
videos. One script replaces the original multi-step pipeline.

#### Output per sample directory

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

#### `--legacy_twostep_mask`

Replicates the original two-step VAE masking pipeline for consistency
with existing checkpoints. See the "Why the flag exists" section below.

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

### `precompute_text_embeddings.py`

Encodes unique prompts from a metadata CSV through the T5 text encoder
and creates per-sample `text_emb.pt` symlinks. See Step 3 above.

### Other flags

| Flag | Description |
|------|-------------|
| `--force` | Re-process directories even if all output files exist |
| `--skip_gpu` | Phase 1 only (directory setup, no GPU encoding) |
| `--also_save_vae_latents` | Save the legacy `vae_latents.pt` alongside `vae_latents_mask_all.pt` |
| `--batch_size N` | Videos per GPU batch for VAE encoding (default 4) |
| `--num_loader_threads N` | I/O prefetch threads (default 4) |
| `--num_cpu_workers N` | CPU workers for Phase 1 directory setup (default 16) |

## Data size estimates (35K combined samples)

| Component | Per sample | Total | Notes |
|-----------|-----------|-------|-------|
| Raw videos (H.264) | ~430 KB | ~12 GB | HDTF/Hallo3 source |
| VoxCeleb2 videos | ~430 KB | ~8.6 GB | Optional (combined list only) |
| audio.wav (PCM 16kHz) | ~230 KB | ~7 GB | |
| Metadata CSV + lists | | <1 MB | |
| **Transfer total** | | **~28 GB** | |
| vae_latents_mask_all.pt | 5.3 MB | ~185 GB | Precomputed on target |
| audio_emb_omniavatar.pt | 7.7 MB | ~270 GB | Precomputed on target |
| ref_latents.pt | 2.7 MB | ~95 GB | Precomputed on target |
| text_emb.pt (8,750 unique) | 8.1 MB | ~69 GB | Precomputed on target |
| **Precomputed total** | ~24 MB | **~620 GB** | |
