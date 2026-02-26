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
│   │   ├── wan_video_vae.py         # Video VAE (unchanged from Wan 2.1)
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
├── scripts/train.py                 # Training script (Accelerate + LoRA + wandb)
├── scripts/inference_modified.py    # Modified inference (no zero audio prefix, matches training)
├── configs/                         # YAML configs (inference_1.3B.yaml, etc.)
├── examples/                        # Sample input files
└── docs/                            # Reference documentation
```

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

## Gotchas

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
