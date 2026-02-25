# OmniAvatar to DiffSynth-Studio Code Mapping

This document maps every OmniAvatar file to its DiffSynth-Studio counterpart.
Status: **ADDED** = new file, **MODIFIED** = forked from DiffSynth with changes, **IDENTICAL** = unchanged copy, **DIFFERENT-PURPOSE** = same filename but substantially different code.

All OmniAvatar paths are relative to `OmniAvatar/` (the package).
All DiffSynth paths are relative to `DiffSynth-Studio/diffsynth/`.

## Pipeline Layer

| OmniAvatar | DiffSynth | Status |
|---|---|---|
| `wan_video.py` | `pipelines/wan_video.py` | MODIFIED |
| `base.py` | `pipelines/base.py` | IDENTICAL |

### `wan_video.py` — MODIFIED

**Added:**
- `log_video()` method: single-chunk denoising with audio support, returns latents for chaining
- `TeaCache` class: token caching for inference acceleration (same code also exists in DiffSynth)
- `audio_emb` dict passed through to DiT in the denoising loop
- Separate `audio_cfg_scale` with 3-forward-pass CFG when it differs from text CFG
- `sp_size`, `sp_group`, `use_unified_sequence_parallel` attributes for sequence parallel
- `denoising_model()` convenience method returning `self.dit`
- `encode_video()` and `decode_video()` helper methods wrapping VAE calls
- Fixed-frame latent overwrite at each denoising step (`latents[:, :, :fixed_frame] = lat[:, :, :fixed_frame]`)

**Removed:**
- `__call__()` method (replaced by `log_video()`)
- `model_fn_wan_video()` external function (SP logic moved into `WanModel.forward()`)
- `encode_control_video()`, `prepare_controlnet_kwargs()`, `prepare_vace_kwargs()`
- Motion controller, VACE, control video, reference latents support
- End-image encoding support in `encode_image()`
- Pipeline units system (`PipelineUnit` classes from `wan_video_new.py`)

**Modified:**
- `from_model_manager()`: adds `infer` param, monkeypatches USP attention into DiT blocks
- `fetch_models()`: fetches only text_encoder, dit, vae, image_encoder (no motion_controller, no VACE)
- `enable_vram_management()`: same pattern but no motion_controller or VACE entries

## Model Layer

| OmniAvatar | DiffSynth | Status |
|---|---|---|
| `models/wan_video_dit.py` | `models/wan_video_dit.py` | MODIFIED |
| `models/wan_video_text_encoder.py` | `models/wan_video_text_encoder.py` | IDENTICAL |
| `models/wan_video_vae.py` | `models/wan_video_vae.py` | IDENTICAL |
| `models/audio_pack.py` | *(none)* | ADDED |
| `models/wav2vec.py` | `models/wav2vec.py` | DIFFERENT-PURPOSE |
| `models/model_manager.py` | `models/model_manager.py` | MODIFIED |

### `models/wan_video_dit.py` — MODIFIED (core change)

This is the most heavily modified file. See `docs/architecture-differences.md` for full details.

**WanModel.__init__:**
- Added: `audio_hidden_size` param, `self.audio_proj` (AudioPack), `self.audio_cond_projs` (ModuleList of Linear)
- Added: global `args` dependency (`args.use_audio`, `args.model_config`)
- Removed: `has_image_pos_emb`, `has_ref_conv`, `add_control_adapter`, `seperated_timestep`, `require_vae_embedding`, `require_clip_embedding`, `fuse_vae_embedding_in_latents` params
- Removed: `self.ref_conv`, `self.control_adapter` modules

**WanModel.forward:**
- Added: `audio_emb`, `tea_cache` parameters
- Added: audio preprocessing (permute, 3-frame prefix, AudioPack, concat projections)
- Added: per-layer audio injection as additive residual
- Added: inline sequence parallel (chunk/pad/all_gather via `args.sp_size`)
- Added: TeaCache check/store/update logic
- Always concatenates `x` and `y` (no conditional `has_image_input` check)
- Does NOT use `clip_feature` / `img_emb` (CLIP embedding unused)

**WanModel.patchify:**
- Simplified: removed `control_adapter` branch
- Returns `(x, grid_sizes)` only

**MLP class:**
- Removed `has_pos_emb` parameter

**WanSelfAttention:**
- Removed `has_image_pos_emb` parameter and related `norm_k_img` logic

**DiTBlock:**
- Constructor: removed `has_image_pos_emb` forwarding
- Forward: removed image positional embedding handling

**WanModelStateDictConverter.from_civitai:**
- Fewer hash entries: only T2V-1.3B, T2V-14B, I2V-14B (no camera control, VACE, TI2V, etc.)
- Added: `args.model_config` override after hash-based detection
- Removed: state_dict prefix filtering for vace/pose/face keys

### `models/audio_pack.py` — ADDED

New module providing the `AudioPack` class:
- Patches audio features with configurable 3D patch size
- Projects via `Linear(in_channels * t * h * w, dim)` + optional `LayerNorm`
- Used with patch_size=[4,1,1], in_channels=10752, dim=32

### `models/wav2vec.py` — DIFFERENT-PURPOSE

Both codebases have a `wav2vec.py` but they serve different models:

| Aspect | OmniAvatar | DiffSynth |
|---|---|---|
| Class name | `Wav2VecModel` | `WanS2VAudioEncoder` |
| Base model | `Wav2Vec2Model` | `Wav2Vec2ForCTC` |
| Pretrained | `wav2vec2-base-960h` | `wav2vec2-large-xlsr-53` |
| Hidden dim | 768 (12 layers) | 1024 (24 layers) |
| Output | Concatenated all hidden states (10752-dim) | Bucketed/interpolated features |
| Purpose | OmniAvatar audio conditioning | Wan 2.2 S2V (speech-to-video) |

### `models/model_manager.py` — MODIFIED

- Added: `infer` param to `ModelManager.__init__()`, `load_model()`, `load()` methods
- Added: import of `smart_load_weights` from io_utils
- Modified: `load_model_from_single_file()` — training mode does xavier_uniform_ init before weight loading; inference mode loads directly
- Removed: `ModelDetectorFromPatchedSingleFile` class (no patched model support, though the class code is still present but unused in configs)
- Removed: LoRA-related model detectors (LoRA handled separately in inference.py)
- Simplified: fewer model_loader_configs entries (only Wan video models)

## Scheduler

| OmniAvatar | DiffSynth | Status |
|---|---|---|
| `schedulers/flow_match.py` | `schedulers/flow_match.py` | IDENTICAL |

## Prompter

| OmniAvatar | DiffSynth | Status |
|---|---|---|
| `prompters/wan_prompter.py` | `prompters/wan_prompter.py` | IDENTICAL |
| `prompters/base_prompter.py` | `prompters/base_prompter.py` | IDENTICAL |

## Distributed / Infrastructure

| OmniAvatar | DiffSynth | Status |
|---|---|---|
| `distributed/xdit_context_parallel.py` | `distributed/xdit_context_parallel.py` | MODIFIED |
| `distributed/fsdp.py` | *(none)* | ADDED |
| `vram_management/` | `vram_management/` | IDENTICAL |

### `distributed/xdit_context_parallel.py` — MODIFIED

- Contains only `usp_attn_forward()` — monkeypatched onto `WanSelfAttention` for distributed attention
- DiffSynth version additionally has `usp_dit_forward()` (a standalone function wrapping the full model forward with SP logic) and `model_fn_wan_video()` — both removed since OmniAvatar puts SP logic inside `WanModel.forward()` directly
- Uses `yunchang.LongContextAttention` for distributed attention

### `distributed/fsdp.py` — ADDED

- `shard_model()` function wrapping DiT blocks with PyTorch FSDP
- Configurable: mixed precision (param/reduce/buffer dtypes), sharding strategy
- Used for memory-efficient multi-GPU inference

## Utilities

| OmniAvatar | DiffSynth | Status |
|---|---|---|
| `utils/args_config.py` | *(none — DiffSynth uses argparse in trainers/utils.py)* | ADDED |
| `utils/audio_preprocess.py` | *(none)* | ADDED |
| `utils/io_utils.py` | `models/utils.py` | MODIFIED |

### `utils/args_config.py` — ADDED

Global `args` singleton pattern:
- `parse_args()` reads YAML config, merges with CLI args, sets as global
- Config reload from `exp_path/config.json` when `reload_cfg=True`
- Hyperparameter override via `-hp key=value,key2=value2`
- Sets distributed env vars: `rank`, `world_size`, `local_rank`, `device`

This singleton is imported by `wan_video_dit.py` at module level — the DiT model depends on `args.use_audio`, `args.sp_size`, and `args.model_config`.

### `utils/audio_preprocess.py` — ADDED

Single function: `add_silence_to_audio_ffmpeg()` — prepends silence to audio using ffmpeg subprocess.

### `utils/io_utils.py` — MODIFIED (from `models/utils.py`)

**Added:**
- `smart_load_weights()`: shape-mismatch-tolerant weight loading — copies checkpoint weights into model params even when shapes differ (model param must be >= checkpoint param in all dims). Critical for loading 16ch base weights into 33ch expanded patch_embedding.
- `save_video_as_grid_and_mp4()`: video saving with optional audio muxing via ffmpeg
- `save_wav()`: audio saving utility
- `init_weights_on_device()`: context manager using `torch.device('meta')` for lazy initialization

**Kept identical:**
- `load_state_dict()`: loads safetensors/bin/ckpt/pth/pt files
- `hash_state_dict_keys()`, `convert_state_dict_keys_to_single_str()`: state dict hashing
- `split_state_dict_with_prefix()`: prefix-based state dict splitting

## Config

| OmniAvatar | DiffSynth | Status |
|---|---|---|
| `configs/model_config.py` | `configs/model_config.py` | MODIFIED |

### `configs/model_config.py` — MODIFIED

- Stripped to only Wan video model hashes (DiT, TextEncoder, VAE, ImageEncoder)
- Removed: all non-Wan model configs (SD, SDXL, FLUX, HunyuanVideo, CogVideo, etc.)
- Removed: preset download configurations
- Kept: `model_loader_configs` list and `huggingface_model_loader_configs` list

## Inference Script (No DiffSynth Counterpart)

| OmniAvatar | Purpose |
|---|---|
| `scripts/inference.py` | End-to-end inference wrapper: model loading, audio encoding, chunked generation loop, video saving |

This script contains the outer loop that calls `pipe.log_video()` per chunk. It is not part of the `OmniAvatar/` package but imports from it.

## DiffSynth Training Infrastructure (Reference for Training Reconstruction)

These files have no OmniAvatar counterpart but are the starting point for training code:

| DiffSynth File | Purpose | Relevance |
|---|---|---|
| `examples/wanvideo/model_training/train.py` | `WanTrainingModule` extending `DiffusionTrainingModule` | Primary training script to adapt |
| `trainers/utils.py` | `DiffusionTrainingModule`, `launch_training_task()`, `wan_parser()` | Training loop infrastructure |
| `trainers/unified_dataset.py` | `UnifiedDataset` with `LoadVideo`, `LoadAudio`, `ImageCropAndResize` operators | Dataset loading — has audio support via `LoadAudio` |
| `pipelines/wan_video_new.py` | Training-capable pipeline with `PipelineUnit` system and `training_loss()` | **This is the training pipeline** (not `wan_video.py`) |

Note: DiffSynth's `wan_video.py` (inference) and `wan_video_new.py` (training) serve different purposes. OmniAvatar's `wan_video.py` is based on the inference version but the training reconstruction should reference `wan_video_new.py` for the loss computation pattern.
