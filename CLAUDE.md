# OmniAvatar — Claude Code Context

## Project Goal

OmniAvatar is an audio-driven portrait animation model built on Wan 2.1. Only inference code
is published. **Our objective is to reconstruct the training code** by bridging OmniAvatar's
inference-only codebase with DiffSynth-Studio's training infrastructure.

**We are working with the 14B model.** The training code must produce identical forward-pass
behavior to inference, with gradients flowing correctly through all components.

## Workflow & Session Rules

- **Context management**: Compact when context reaches ~60%. Use subagents (Task tool
  with Explore type) for all code exploration to keep the main context clean.
- **Environment**: `conda activate omniavatar`
- **GPUs**: We have GPUs 0 and 1 available. Use `CUDA_VISIBLE_DEVICES=0,1`.
- **Write scope**: Full read/write access to everything within this repo
  (`/home/work/.local/OmniAvatar/`). Do NOT write to files outside this repo.
- **Version control**: Commit and push after verified large changes so we can always
  roll back. Don't let uncommitted work accumulate across major milestones.
- After a productive session with significant code changes, run `/revise-claude-md` to
  capture new insights into this file.
- Periodically ask "check if my CLAUDE.md is up to date" to trigger a full audit.

## Repository Layout

```
OmniAvatar/                          # Root
├── OmniAvatar/                      # Main package (inference code)
│   ├── wan_video.py                 # WanVideoPipeline (log_video method)
│   ├── base.py                      # BasePipeline
│   ├── models/
│   │   ├── wan_video_dit.py         # DiT model — core changes here
│   │   ├── wan_video_text_encoder.py  # T5 text encoder (unchanged from Wan 2.1)
│   │   ├── wan_video_vae.py         # Video VAE (+ gradient checkpointing for aux losses)
│   │   ├── audio_pack.py            # AudioPack module (NEW)
│   │   ├── wav2vec.py               # Wav2Vec2 audio encoder (NEW)
│   │   └── model_manager.py         # Model loading + smart_load_weights
│   ├── configs/model_config.py      # Model hash detection configs
│   ├── prompters/                   # T5 text prompting (unchanged)
│   ├── schedulers/flow_match.py     # Flow matching scheduler (unchanged)
│   ├── distributed/                 # FSDP + sequence parallel
│   ├── utils/
│   │   ├── args_config.py           # Global args singleton (YAML + CLI)
│   │   ├── io_utils.py              # Weight loading, smart_load_weights
│   │   └── audio_preprocess.py      # Audio silence padding
│   └── vram_management/             # CPU offloading (unchanged)
├── DiffSynth-Studio/                # Reference codebase (has training code)
│   └── diffsynth/
│       ├── pipelines/wan_video_new.py  # Training-capable pipeline w/ training_loss()
│       ├── trainers/utils.py        # DiffusionTrainingModule, launch_training_task
│       ├── trainers/unified_dataset.py # UnifiedDataset w/ LoadVideo, LoadAudio
│       └── models/, schedulers/, etc.
├── scripts/inference.py             # Inference entry point (chunked generation loop)
├── scripts/train.py                 # I2V training script (Accelerate + LoRA + wandb)
├── scripts/train_v2v.py             # V2V lip sync training (49ch input, spatial mask)
├── scripts/precompute_audio_omniavatar.py  # Audio precomputation (10752-dim embeddings)
├── scripts/train_v2v_auxloss.sh     # Launch script for V2V training with aux losses
├── scripts/inference_modified.py    # Modified inference (no zero audio prefix, matches training)
├── configs/                         # YAML configs (inference_1.3B.yaml, etc.)
├── examples/                        # Sample input files
└── docs/                            # Reference documentation
```

## Training Data

- **I2V base path**: `/home/work/.local/combined_data/high_visual_quality` (NOT `/home/work/combined_data/`)
- **I2V metadata**: `test_metadata.csv` (in repo root, 3 test samples for smoke tests)
- **V2V training**: `/home/work/stableavatar_data/v2v_training_data/` (29K videos)
  - Path list: `video_square_path.txt` (one directory per line)
  - Each dir has: `sub_clip.mp4`, `audio.wav`, `prompt.txt`, `vae_latents.pt`, `audio_emb_omniavatar.pt`, `text_emb.pt`
- **V2V validation**: `/home/work/stableavatar_data/v2v_validation_data/{recon,mixed}/`
  - Recon: 10 samples (same identity video + audio), Mixed: 12 samples (cross-identity)
- **LatentSync mask**: `/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png`

## Running Training

```bash
# Single GPU (use env binaries directly, not conda run — it swallows multi-process output)
CUDA_VISIBLE_DEVICES=0 /home/work/.local/miniconda3/envs/omniavatar/bin/accelerate launch \
  --num_processes 1 scripts/train.py \
  --dit_paths "pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-0000{1..6}-of-00006.safetensors" \
  --text_encoder_path pretrained_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth \
  --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
  --wav2vec_path pretrained_models/wav2vec2-base-960h \
  --omniavatar_ckpt pretrained_models/OmniAvatar-14B/pytorch_model.pt \
  --dataset_base_path /path/to/videos --dataset_metadata_path metadata.csv \
  --use_gradient_checkpointing --num_frames 81 --height 512 --width 512

# 2 GPUs with gradient accumulation
CUDA_VISIBLE_DEVICES=0,1 /home/work/.local/miniconda3/envs/omniavatar/bin/accelerate launch \
  --config_file configs/accelerate_2gpus.yaml scripts/train.py [same args] \
  --gradient_accumulation_steps 4
```

## Running V2V Training

```bash
# V2V with aux losses (single GPU, precomputed data)
bash scripts/train_v2v_auxloss.sh

# V2V without aux losses (no offloading needed, ~61 GB peak)
CUDA_VISIBLE_DEVICES=0 /home/work/.local/miniconda3/envs/omniavatar/bin/accelerate launch \
  --num_processes 1 --mixed_precision bf16 scripts/train_v2v.py \
  [model paths] --data_list_path /path/to/video_square_path.txt \
  --latentsync_mask_path /path/to/mask.png \
  --use_precomputed_vae --use_precomputed_audio --use_precomputed_text_emb \
  --use_gradient_checkpointing

# Precompute OmniAvatar audio embeddings (2 GPUs parallel)
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_audio_omniavatar.py \
  --wav2vec_path pretrained_models/wav2vec2-base-960h \
  --data_list_path /path/to/video_square_path.txt --shard_id 0 --num_shards 2 &
CUDA_VISIBLE_DEVICES=1 python scripts/precompute_audio_omniavatar.py \
  --wav2vec_path pretrained_models/wav2vec2-base-960h \
  --data_list_path /path/to/video_square_path.txt --shard_id 1 --num_shards 2 &
wait
```

## Running Inference

```bash
# 14B single-GPU
torchrun --standalone --nproc_per_node=1 scripts/inference.py \
  --config configs/inference.yaml --input_file examples/infer_samples.txt

# 1.3B single-GPU
torchrun --standalone --nproc_per_node=1 scripts/inference.py \
  --config configs/inference_1.3B.yaml --input_file examples/infer_samples.txt

# 14B multi-GPU (8x)
torchrun --standalone --nproc_per_node=8 scripts/inference.py \
  --config configs/inference.yaml --input_file examples/infer_samples.txt \
  --hp=sp_size=8,max_tokens=30000,use_fsdp=True
```

**Input file format** (`@@`-delimited): `prompt@@image_path@@audio_path` (one sample per line).

## How Inference Works

1. `args_config.parse_args()` loads YAML config + CLI args into global `args` singleton
2. `ModelManager` loads base Wan T2V weights (DiT, T5 text encoder, VAE) from safetensors
3. `WanVideoPipeline.from_model_manager()` creates pipeline; monkeypatches USP attention if multi-GPU
4. OmniAvatar checkpoint loaded: either full weights via `load_state_dict(strict=True)` or LoRA via PEFT
5. Wav2Vec2 encodes audio → concatenates all hidden states → 10752-dim features per frame
6. Sliding-window chunk loop: each chunk has `L` frames with `overlap_frame` overlap
   - Reference image VAE-encoded, repeated across time, mask concatenated → `y` input (17ch)
   - Audio sliced per chunk with prefix carry-forward for continuity
   - `pipe.log_video()` runs denoising: text+audio CFG, fixed-frame overwrite, TeaCache
   - Previous chunk's tail latent becomes next chunk's prefix
7. Chunks stitched and trimmed to audio length; decoded via VAE

## Key Architecture Change: Audio Conditioning

OmniAvatar adds audio conditioning to the Wan 2.1 DiT:
- **Wav2Vec2** (`wav2vec2-base-960h`) extracts 768-dim features, all 14 hidden states concatenated → 10752-dim
- **AudioPack** patches with [4,1,1], projects 10752→32-dim with LayerNorm
- **Per-layer injection**: `num_layers//2 - 1` linear projections (32→model_dim), applied as additive
  residuals to DiT layers 2 through num_layers//2 only
- **Audio CFG**: separate scale from text CFG; when different, uses 3 forward passes

## Reference Documents

Detailed documentation for training code reconstruction:
- **`docs/architecture-differences.md`** — All OmniAvatar vs Wan 2.1 differences: audio pipeline,
  DiT changes, pipeline changes, patch embedding expansion, training-relevant details
- **`docs/code-mapping.md`** — File-by-file mapping between OmniAvatar and DiffSynth-Studio
  with status (ADDED/MODIFIED/IDENTICAL) and specific change notes per file

## Important Notes for Training Reconstruction

1. **Global `args` singleton**: `wan_video_dit.py` reads `args.use_audio`, `args.sp_size`,
   `args.model_config` from the global `args` object in `utils/args_config.py`. Training code
   must either maintain this pattern or refactor it out.

2. **T2V base, not I2V**: OmniAvatar loads T2V weights (in_dim=16, has_image_input=False).
   The I2V-style reference frame input is NOT via the standard Wan I2V path.

3. **No CLIP encoder**: `image_encoder_path: None` in all configs. Reference frames are handled
   entirely via VAE encoding + binary mask concatenation as the `y` input to the DiT.

4. **Patch embedding expansion**: Base T2V has 16ch input. OmniAvatar concatenates noise (16ch) +
   ref latent (16ch) + mask (1ch) = 33ch. This is done via `args.model_config` overriding `in_dim`,
   with `smart_load_weights()` copying 16ch base weights into the first 16ch of 33ch.

5. **Audio on middle layers only**: Injection at layers where `layer_i > 1` and
   `layer_i <= num_layers // 2` (0-indexed). Layers 0-1 and the second half get no audio.

6. **DiffSynth training pipeline**: Use `wan_video_new.py` (not `wan_video.py`) — it has
   `training_loss()` and the `PipelineUnit` system. The training script is
   `DiffSynth-Studio/examples/wanvideo/model_training/train.py`.

7. **Weight loading for training**: `model_manager.py` line 27-33 — training mode does
   `xavier_uniform_(gain=0.05)` init, then `smart_load_weights()` overlays base weights.
   This ensures new parameters (audio modules, expanded patch_embedding) start initialized.

8. **Checkpoint `config.json`**: Each trained checkpoint (e.g., `pretrained_models/OmniAvatar-14B/config.json`)
   stores the training config. When `reload_cfg: True` in YAML, this is loaded into the `args` singleton,
   providing `model_config: {in_dim: 33}`, `train_architecture: lora`, `lora_rank: 128`,
   `lora_alpha: 64`, `use_audio: true`, `i2v: true`, `random_prefix_frames: true`, etc.

## Condition Dropout & Auxiliary Losses (train.py)

**Condition dropout**: `--text_drop_prob 0.1 --audio_drop_prob 0.1` — randomly replaces
text with empty encoding or audio with zeros during training for CFG support.

**Auxiliary losses** on VAE-decoded x_0 predictions (each independently enabled):
- `--use_sync_loss`: StableSyncNet chunked cosine-similarity BCE (lower half, 16-frame chunks)
- `--use_lpips_loss`: VGG LPIPS perceptual loss (lower half)
- `--use_trepa_loss`: VideoMAE-ViT-G temporal feature matching
- `--use_vae_gradient_checkpointing`: Required for memory when using aux losses
- x_0 formula: `x_0 = x_t - sigma * noise_pred`, clamped [-10, 10]
- Defaults from StableAvatar: `aux_recon=1.0, sync=0.1, lpips=0.1, trepa=10.0`

**External deps** (imported via sys.path from `/home/work/.local/Self-Forcing_LipSync_StableAvatar/`):
StableSyncNet, melspectrogram (audio.py), TREPALoss, lpips pip package

## V2V Training (train_v2v.py)

- **49-channel input**: 16ch noise + 16ch ref_repeated + 1ch spatial_mask + 16ch masked_video.
  `in_dim=49` in args singleton (vs 33 for I2V train.py).
- **Precomputed data**: `vae_latents.pt` from StableAvatar (compatible Wan VAE),
  `audio_emb_omniavatar.pt` from our precompute script (10752-dim, NOT StableAvatar's 768-dim).
- **Precomputed text**: `text_emb.pt` from StableAvatar preprocessing (same T5 model, [1, 512, 4096]).
  `--use_precomputed_text_emb` skips T5 loading entirely — saves ~250-450ms/step with offloading.
- **Patch embedding expansion**: OmniAvatar ckpt has 33ch, model is 49ch. Constructor
  handles expansion: copies 33ch weights, leaves channels 33-48 xavier-initialized.
- **LatentSync mask**: 256x256 PNG (255=keep upper face, 0=mask mouth). Resized to latent
  resolution (64x64) via bilinear interpolation + threshold. Inverted to OmniAvatar
  convention (0=keep, 1=generate) before use.

## VRAM Budget (14B model, 512x512x81, single H200 150GB)

| Mode | DiT | T5 | VAE+W2V | Optimizer | Activations | Peak |
|------|-----|----|---------|-----------|-------------|------|
| No aux losses | 30 GB | 11 GB | 0.6 GB | 3.7 GB | ~17 GB | **~61 GB** |
| With aux losses | 30 GB | 11 GB | 0.6 GB | 3.7 GB | ~100 GB | **~146 GB** |
| Aux + offload_frozen | 30 GB | CPU | 0.6 GB | 3.7 GB | ~100 GB | **~134 GB** |

- **`--offload_frozen` required with aux losses (unless precomputed text+audio)** — offloads
  T5 (11.36 GB) + Wav2Vec (0.38 GB) to CPU. VAE stays on GPU (0.26 GB, needed for backward
  through aux losses). With precomputed text+audio, `--offload_frozen` is unnecessary — T5
  and Wav2Vec2 are never loaded to GPU, saving per-step offloading overhead.
- **VAE decode is the VRAM spike**: decoding 21 latent frames → 81 RGB@512x512 costs ~50 GB.
- **`--verbose_vram`**: prints VRAM at each stage of forward pass for debugging.

## Gotchas

- **`dit.training=False` after accelerator.prepare()** — critical bug we fixed. The DiT's
  gradient checkpointing checks `self.training` (wan_video_dit.py:405). Without explicit
  `.train()`, it's silently False → 105+ GB activations instead of ~17 GB. Fixed via
  `_set_training_modes()` (DiffSynth's `freeze_except` pattern): `.train()` on DiT,
  `.eval()` on frozen components. Called in `__init__`, after `prepare()`, and defensively
  in `forward()`.
- **Don't offload VAE to CPU when using aux losses** — backward needs VAE weights on GPU
  for gradient flow through the `with_grad=True` decode path.
- **Don't use `conda run` for multi-process accelerate** — it swallows output. Use
  `/home/work/.local/miniconda3/envs/omniavatar/bin/accelerate` directly.
- **Wav2Vec2 must stay float32** — inference doesn't cast it to bf16, and the CNN
  feature extractor fails on bf16 input.
- **Audio must be trimmed to match video frame count** — if the source video is longer
  than `num_frames`, `linear_interpolation` in Wav2Vec2 will compress all audio into
  fewer frames, destroying sync.
- **`pipe.device` is not updated by `model.to(device)`** — must set `pipe.device`
  explicitly when using VAE encode/decode outside the Accelerate context.
- **`model_manager.py` training init uses CPU** (we changed from hardcoded `cuda`) —
  needed for multi-GPU DDP where each rank must init on its own device.
- **System Python 3.12 packages can contaminate conda** — if imports fail with
  `av._core` errors, install `pyav` explicitly in the conda env.
- **WanVideoVAE.decode() moves tensors to CPU** — breaks gradient flow for aux losses.
  Use `pipe.vae.model.decode(latents, pipe.vae.scale)` directly + non-inplace `clamp(-1, 1)`.
- **SyncNet expects exactly chunk_size (16) frames** — partial chunks cause channel mismatch.
  Skip chunks with fewer than chunk_size frames.
- **VAE `requires_grad_(False)` does NOT block gradient flow** — it only prevents gradient
  accumulation on VAE params. Autograd still flows through for aux loss backprop to DiT.
- **Don't use `find_unused_parameters=True` in DDP** — unnecessary for our LoRA+audio setup
  (condition dropout zeros inputs but modules still execute). Adds expensive autograd graph
  traversal after every backward. DiffSynth defaults to False; StableAvatar doesn't use it.
- **Avoid `.item()` in forward()** — each call forces CUDA synchronization. Store as
  `.detach()` tensors, call `.item()` only at logging time.
- **Use precomputed text with `--offload_frozen`** — without precomputed text, T5 (11 GB)
  transfers GPU↔CPU every step plus `empty_cache()`. Biggest fixable perf bottleneck.
- **xfuser conflicts with Flash Attention 3** — xfuser registers simplified flash_attn_3
  operators that break real FA3 ("expected 3 arguments but received 32"). Import
  `flash_attn_interface` BEFORE xfuser. Training (sp_size=1) uses FA3; multi-GPU
  inference (sp_size>1) falls back to FA2 automatically.

## Model Configurations

| Parameter | 14B | 1.3B |
|---|---|---|
| dim | 5120 | 1536 |
| ffn_dim | 13824 | 8960 |
| num_heads | 40 | 12 |
| num_layers | 40 | 30 |
| audio_hidden_size | 32 | 32 |
| audio_cond_projs | 19 | 14 |
| num_inference_steps | 25 or 50 | 50 |
| overlap_frame | 13 | 13 |
| guidance_scale | 4.5 | 4.5 |
