# OmniAvatar vs Wan 2.1: Architecture Differences

This document details every modification OmniAvatar makes to the Wan 2.1 video generation model.
It is intended as a reference for reconstructing OmniAvatar training code.

## 1. Overview

OmniAvatar transforms Wan 2.1 T2V (text-to-video) into an audio-driven portrait animation system.
The key changes are:

1. **Audio conditioning** via Wav2Vec2 + AudioPack + per-layer injection into the DiT
2. **Reference frame handling** via VAE encoding + mask concatenation (no CLIP encoder)
3. **Expanded patch embedding** from 16ch to 33ch input
4. **Chunked sliding-window inference** for arbitrary-length video
5. **Inference optimizations**: TeaCache, sequence parallel, FSDP, VRAM offloading
6. **Removed features**: CLIP image encoder, motion controller, VACE, control adapters, end-image support

## 2. Audio Conditioning Pipeline

This is the most significant architectural addition.

### 2.1 Wav2Vec2 Audio Encoder

- **Model**: `wav2vec2-base-960h` (`Wav2Vec2Model`, NOT `Wav2Vec2ForCTC`)
- **File**: `OmniAvatar/models/wav2vec.py` — custom `Wav2VecModel` extending HuggingFace's `Wav2Vec2Model`
- **Input**: raw waveform at 16kHz
- **Architecture**: 12 transformer encoder layers, 768-dim hidden states
- **Custom modification**: `linear_interpolation()` resizes the feature sequence to match the target video frame count (`seq_len`) using `F.interpolate(mode='linear')`
- **Output**: `last_hidden_state` (B, T, 768) + `hidden_states` tuple of 13 tensors (feature extractor output + 12 encoder layers)

**Feature concatenation** (in `scripts/inference.py:250-254`):
```python
audio_embeddings = hidden_states.last_hidden_state          # (B, T, 768)
for mid_hidden_states in hidden_states.hidden_states:       # 13 items, each (B, T, 768)
    audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
# Result: (B, T, 10752)  — 768 * 14 = 10752 (last_hidden_state is duplicated with hidden_states[-1])
```

### 2.2 AudioPack Module

- **File**: `OmniAvatar/models/audio_pack.py`
- **Purpose**: Patches and projects audio features into a compact representation
- **Patch size**: `[4, 1, 1]` (4 temporal frames, 1x1 spatial)
- **Input dim**: `10752 * 4 * 1 * 1 = 43008` (after rearrange into patches)
- **Output dim**: 32 (`audio_hidden_size` parameter)
- **Layers**: `Linear(43008, 32)` + optional `LayerNorm(32)`

```python
# In AudioPack.forward():
vid = rearrange(vid, "b c (T t) (H h) (W w) -> b T H W (t h w c)", t=4, h=1, w=1)
vid = self.proj(vid)  # Linear(43008, 32)
vid = self.norm_out(vid)  # LayerNorm(32)
```

### 2.3 Silence Prefix Padding

Before AudioPack, 3 frames of silence are prepended by repeating the first audio frame:

```python
# wan_video_dit.py:352
audio_emb = torch.cat([audio_emb[:, :, :1].repeat(1, 1, 3, 1, 1), audio_emb], 2)
```

This means T_audio frames become T_audio + 3 before patching. After AudioPack with patch_size=4, the temporal dimension becomes `(T_audio + 3) // 4`.

### 2.4 Per-Layer Audio Projection

- **Number of projection heads**: `num_layers // 2 - 1` (19 for 14B with 40 layers; 14 for 1.3B with 30 layers)
- **Each head**: `Linear(32, dim)` where dim is the model dimension (5120 for 14B, 1536 for 1.3B)
- **All projections computed at once**: `torch.concat([proj(audio_emb) for proj in self.audio_cond_projs], 0)`
  - This stacks N projections along batch dim: shape `(N*B, T_patched, H_audio, W_audio, dim)`

### 2.5 Audio Injection into DiT Blocks

Audio is injected as an additive residual into selected transformer layers only:

```python
# wan_video_dit.py:391-403
for layer_i, block in enumerate(self.blocks):
    if self.use_audio:
        if (layer_i <= len(self.blocks) // 2 and layer_i > 1):
            au_idx = layer_i - 2
            audio_emb_tmp = audio_emb[:, au_idx].repeat(1, 1, lat_h // 2, lat_w // 2, 1)
            audio_cond_tmp = self.patchify(audio_emb_tmp.permute(0, 4, 1, 2, 3))[0]
            x = audio_cond_tmp + x    # additive residual
```

**Which layers receive audio**: layers 2 through `num_layers // 2` (inclusive).
For 14B (40 layers): layers 2-20 = 19 layers.
For 1.3B (30 layers): layers 2-15 = 14 layers.
(Layer indexing is 0-based.)

**Spatial tiling**: Audio features are repeated to match the downsampled latent spatial dims: `repeat(1, 1, lat_h // 2, lat_w // 2, 1)`, then `patchify()` converts to the sequence dimension matching the latent tokens.

### 2.6 Audio CFG (Classifier-Free Guidance)

OmniAvatar supports a separate `audio_cfg_scale` independent of text `cfg_scale`.

**When `audio_cfg_scale == cfg_scale`** (default): 2 forward passes
```python
noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
# where noise_pred_nega uses negative text + zero audio
```

**When `audio_cfg_scale != cfg_scale`**: 3 forward passes
```python
text_noise_pred_nega  = dit(latents, negative_text, zero_audio)    # text-unconditional
audio_noise_pred_nega = dit(latents, positive_text, zero_audio)    # audio-unconditional
noise_pred_posi       = dit(latents, positive_text, real_audio)    # fully conditioned

noise_pred = text_noise_pred_nega
           + cfg_scale * (audio_noise_pred_nega - text_noise_pred_nega)
           + audio_cfg_scale * (noise_pred_posi - audio_noise_pred_nega)
```

(See `OmniAvatar/wan_video.py:262-273`.)

## 3. DiT Model Changes

**File**: `OmniAvatar/models/wan_video_dit.py`

### 3.1 Constructor Changes

**Added parameters:**
- `audio_hidden_size: int = 32` — dimension of AudioPack output

**Added modules** (conditional on `args.use_audio`):
- `self.audio_proj`: `AudioPack(10752, [4,1,1], 32, layernorm=True)`
- `self.audio_cond_projs`: `ModuleList` of `num_layers//2 - 1` `Linear(32, dim)` layers

**Removed parameters** (present in DiffSynth, absent in OmniAvatar):
- `has_image_pos_emb`, `has_ref_conv`, `add_control_adapter`, `in_dim_control_adapter`
- `seperated_timestep`, `require_vae_embedding`, `require_clip_embedding`, `fuse_vae_embedding_in_latents`

**Global `args` dependency**: The constructor reads `args.use_audio` and `args.model_config` from the global singleton defined in `OmniAvatar/utils/args_config.py`. This is a design concern for training code — the DiT model initialization depends on runtime args.

### 3.2 Forward Method Changes

**Added parameters**: `audio_emb`, `tea_cache`

**Key behavioral differences from DiffSynth:**
1. Always concatenates `y` (reference image latent): `x = torch.cat([x, y], dim=1)` — no conditional `has_image_input` check
2. Does NOT use CLIP embedding — no `clip_feature` processing even though the parameter exists in the signature
3. Sequence parallel logic is inline in forward (chunking, padding, all_gather) rather than in a separate `model_fn_wan_video()` function
4. Audio preprocessing and per-layer injection happen inside forward
5. TeaCache check/store/update logic integrated directly

### 3.3 Patch Embedding Expansion

The base Wan T2V model has `in_dim=16` (noise latent channels). OmniAvatar concatenates:
- Noise latent: 16 channels
- Reference image latent (repeated across time): 16 channels
- Binary mask (0=known frame, 1=to generate): 1 channel

Total: **33 channels** input to `patch_embedding` (a `Conv3d(33, dim, kernel_size=patch_size, stride=patch_size)`).

This expansion is handled via `args.model_config` which overrides the auto-detected config's `in_dim` from 16 to 33. The `smart_load_weights()` function then handles the shape mismatch: it copies the 16-channel base weights into the first 16 channels of the 33-channel weight tensor, leaving the rest at their initialized values (xavier uniform for training, or zero for inference).

### 3.4 DiTBlock Simplification

OmniAvatar's `DiTBlock` removes:
- `has_image_pos_emb` positional embedding support in `WanSelfAttention`
- Image-specific cross-attention key norm (`norm_k_img`) — only text cross-attention remains

The `MLP` class also drops the `has_pos_emb` parameter.

### 3.5 Model Config Detection

`WanModelStateDictConverter.from_civitai()` uses state dict key hashing to auto-detect model variant:
- `9269f8db...` → 1.3B T2V (dim=1536, 30 layers, in_dim=16)
- `aafcfd96...` → 14B T2V (dim=5120, 40 layers, in_dim=16)
- `6bfcfb3b...` → 14B I2V (dim=5120, 40 layers, in_dim=36)

After hash-based detection, `args.model_config` can override any field (critical for setting `in_dim=33`).

## 4. Pipeline Changes

**File**: `OmniAvatar/wan_video.py`

### 4.1 `log_video()` Replaces `__call__()`

OmniAvatar's pipeline exposes `log_video()` instead of `__call__()`:
- Takes a pre-encoded latent (`lat`) rather than raw image/text inputs
- Takes `audio_emb` dict directly
- Returns `(frames, recons, latents)` when `return_latent=True` — the latents are needed for the next chunk
- Always runs CFG (no unconditional-only path except when `cfg_scale == 1.0`)

### 4.2 Fixed-Frame Latent Overwrite

During denoising, the first `fixed_frame` latent frames are overwritten each step:
```python
if fixed_frame > 0:
    latents[:, :, :fixed_frame] = lat[:, :, :fixed_frame]
```
This anchors the reference frame(s) and ensures temporal continuity between chunks.

### 4.3 Removed Features

Compared to DiffSynth's `WanVideoPipeline`:
- No `motion_controller` / motion guidance
- No VACE (Video Appearance Control Engine)
- No `control_video` / controlnet conditioning
- No `end_image` support (only start/reference frame)
- No `model_fn_wan_video()` function — model call is direct

## 5. Chunked Sliding-Window Inference

**File**: `scripts/inference.py:261-320`

OmniAvatar generates long videos by splitting into overlapping chunks:

1. **Calculate chunk parameters**: `L` frames per chunk (from `max_tokens` / resolution), `overlap_frame` (default 13, must be 1+4n)
2. **First chunk**: `first_fixed_frame=1` (the reference image frame)
3. **Subsequent chunks**: `fixed_frame=overlap_frame` frames from the previous chunk's tail
4. **Audio slicing**: each chunk gets its corresponding audio segment + `audio_prefix` (last `fixed_frame` audio frames from previous chunk)
5. **Video assembly**: `torch.cat(video, dim=1)`, trimmed to original audio length

The chunking loop lives in `inference.py`, NOT in the pipeline — `log_video()` handles one chunk at a time.

## 6. Reference Frame Handling

**No CLIP encoder is used.** Both config YAMLs set `image_encoder_path: None`.

Reference frame conditioning works entirely through VAE:
1. Reference image encoded via `pipe.encode_video()` → latent of shape `(1, 16, T_lat, H_lat, W_lat)` where `T_lat = 1` for one frame
2. Latent repeated across time: `img_lat.repeat(1, 1, T, 1, 1)` → 16 channels
3. Binary mask created: `msk[:, :, 1:] = 1` (0 for first frame = known, 1 for rest = generate) → 1 channel
4. Concatenated as `image_emb["y"] = torch.cat([image_cat, msk], dim=1)` → 17 channels
5. In DiT forward: `x = torch.cat([x, y], dim=1)` → 16 (noise) + 17 (ref + mask) = 33 channels

## 7. Training-Relevant Summary

### Trainable Audio Modules

These are the OmniAvatar-specific parameters NOT present in base Wan 2.1 weights:
- `audio_proj` (AudioPack): `Linear(43008, 32)` + `LayerNorm(32)` — 1.38M params
- `audio_cond_projs` (ModuleList): N x `Linear(32, dim)` — N * 32 * dim params
  - 14B: 19 x `Linear(32, 5120)` = 3.11M params
  - 1.3B: 14 x `Linear(32, 1536)` = 0.69M params
- `patch_embedding` expansion: extra 17 input channels worth of Conv3d weights

### LoRA Configuration

From `pretrained_models/OmniAvatar-14B/config.json`:
- **Target modules**: `q,k,v,o,ffn.0,ffn.2` (all attention projections + FFN)
- **LoRA rank**: 128
- **LoRA alpha**: 64
- **Init weights**: `kaiming` (mapped to `True` in PEFT's `init_lora_weights`)
- **Library**: PEFT `inject_adapter_in_model()`

### Weight Initialization (Training Mode)

From `model_manager.py:27-33`:
```python
if not infer:  # training
    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param, gain=0.05)
        else:
            nn.init.zeros_(param)
```
Then base weights are loaded via `smart_load_weights()` (overwriting the xavier init for matching parameters).

### Gradient Checkpointing

Supported in DiT forward via `use_gradient_checkpointing` and `use_gradient_checkpointing_offload` flags. The offload variant uses `torch.autograd.graph.save_on_cpu()`.

### Loss Function

OmniAvatar itself does not contain a training loss function. DiffSynth's `wan_video_new.py` has `training_loss()` which should be adapted. The flow matching scheduler's loss is: predict noise from noised latent at random timestep, MSE between predicted and target noise.
