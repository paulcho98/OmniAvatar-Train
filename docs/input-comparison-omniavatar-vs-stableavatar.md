# Training Input Comparison: OmniAvatar vs StableAvatar V2V

This document compares how training inputs are constructed in OmniAvatar's training
script versus StableAvatar's V2V lip sync training pipeline. The focus is on the
conditioning input `y`, masks, reference frames, and the presence of masked video.

---

## Quick Summary

| Aspect | OmniAvatar | StableAvatar V2V |
|--------|-----------|-----------------|
| **Base model** | Wan 2.1 T2V (16ch input) | Wan 2.1 Fun InP (I2V, 36ch input) |
| **Total patch_embed channels** | 33 | 52 (36 base + 16 ref expansion) |
| **y breakdown** | 16ch ref + 1ch mask = 17ch | 4ch mask + 16ch masked_video + 16ch ref_seq = 36ch |
| **Masked video** | No | Yes (mouth region zeroed via LatentSync mask) |
| **Mask type** | Binary temporal (1ch) | Spatial mouth mask, temporally grouped (4ch) |
| **Reference source** | First frame of training clip | Separate 81-frame segment from same video |
| **Reference format** | Single frame repeated across time | Full video segment VAE-encoded |
| **CLIP image features** | Not used | Yes (reference frame) |
| **Audio dim** | 10752 (14 Wav2Vec2 hidden states) | 768 (last_hidden_state only) |
| **Audio injection** | Additive residual at layers 2–20 | Vocal projector throughout transformer |

---

## 1. The `y` Conditioning Tensor

### OmniAvatar: `y = [ref_latent_repeated, binary_mask]` — 17 channels

**Code:** `scripts/train.py:366-371`

```python
def prepare_reference_input(ref_latent, num_lat_frames):
    # ref_latent: (1, 16, 1, H_lat, W_lat) — first frame of GT video
    image_cat = ref_latent.repeat(1, 1, num_lat_frames, 1, 1)  # (1, 16, T_lat, H, W)
    msk = torch.zeros_like(image_cat[:, :1])                    # (1, 1, T_lat, H, W)
    msk[:, :, 1:] = 1                                           # 0 at t=0, 1 at t>0
    return torch.cat([image_cat, msk], dim=1)                   # (1, 17, T_lat, H, W)
```

**Channel layout of y (17ch):**
- Channels 0–15: VAE-encoded first frame, **repeated identically** at every latent timestep
- Channel 16: binary mask — `0` marks the reference frame (t=0), `1` marks frames to generate (t>0)

**Then x + y concatenation** (`wan_video_dit.py:357`):
```python
x = torch.cat([x, y], dim=1)  # (1, 16+17, T, H, W) = (1, 33, T, H, W)
```

### StableAvatar V2V: `y = [mask_4ch, masked_latents, ref_seq_latents]` — 36 channels

**Code:** `train_1B_square.py:1643-1663`

**Step 1 — Temporal mask grouping (lines 1644–1651):**
```python
# pixel_value_masks starts as: (B, 1, 81, H, W)
#   Frame 0: all 0s
#   Frames 1–80: inverted LatentSync mask (1=mouth, 0=upper face)

# Repeat frame 0 four times + concat frames 1+
pixel_value_masks = concat([
    repeat_interleave(pvm[:, :, 0:1], repeats=4, dim=2),  # (B, 1, 4, H, W)
    pvm[:, :, 1:]                                           # (B, 1, 80, H, W)
], dim=2)                                                   # (B, 1, 84, H, W)

# Group into 4-frame patches (matching VAE 4× temporal compression)
pvm = pvm.view(B, 84//4, 4, H, W)   # (B, 21, 4, H, W)
pvm = pvm.transpose(1, 2)            # (B, 4, 21, H, W)
```

**Step 2 — Invert and resize to latent space (line 1652):**
```python
pixel_value_masks = resize_mask(1 - pixel_value_masks, latents)
# Output: (B, 4, 21, 64, 64)
# After inversion: 1 = keep (upper face), 0 = generate (mouth)
```

**Step 3 — Concatenate (lines 1659–1662):**
```python
inpaint_latents = torch.concat([pixel_value_masks, masked_latents], dim=1)
# (B, 4+16, 21, 64, 64) = (B, 20, 21, 64, 64)

# With --use_precomputed_ref:
ref_latents = batch["precomputed_ref_latents"]           # (B, 16, 21, 64, 64)
inpaint_latents = torch.concat([inpaint_latents, ref_latents], dim=1)
# (B, 20+16, 21, 64, 64) = (B, 36, 21, 64, 64)
```

**Channel layout of y (36ch):**
- Channels 0–3: Spatial mask, temporally grouped (4 mask frames per latent timestep)
  - Frame 0 group: all 1s (keep entire first frame)
  - Other groups: LatentSync mask values (1=keep upper face, 0=generate mouth)
- Channels 4–19: **Masked video latents** — VAE encoding of video with mouth region zeroed out
- Channels 20–35: **Reference sequence latents** — VAE encoding of a different 81-frame segment

**Then x + y concatenation:**
```python
noise_pred = transformer3d(x=noisy_latents, y=inpaint_latents, ...)
# Inside transformer: x = cat([x, y], dim=channel) → (B, 16+36, T, H, W) = (B, 52, T, H, W)
```

---

## 2. The Mask

### OmniAvatar: Binary Temporal Mask (1 channel)

```
t=0: 0000000000    (this is the reference frame — "known")
t=1: 1111111111    (generate this frame)
t=2: 1111111111    (generate this frame)
...
```

- **1 channel**, uniform across spatial dimensions
- Only distinguishes "reference frame" (t=0) from "frames to generate" (t>0)
- **No spatial information** — doesn't tell the model which image region to reconstruct
- The model must figure out what to change and what to preserve from the audio signal alone

### StableAvatar V2V: Spatial Mouth Mask (4 channels, from LatentSync)

```
t=0: 1111111111    (keep entire frame — identity anchor)
t>0: 1110001111    (keep upper face=1, generate mouth=0)
     1110001111
     1111111111
```

- **4 channels** due to temporal grouping (4 original video frames per latent timestep)
- **Spatially varying** — a fixed mouth-shaped region from a LatentSync mask PNG
- Loaded once at init: `mask_img = Image.open(mask_path).convert('L'); mask = np.array(mask_img) / 255.0`
  - Original mask convention: `1=keep (upper face), 0=mask (mouth)`
- Frame 0 is always fully unmasked (all 1s after inversion) — serves as identity anchor
- Explicitly tells the model: "reconstruct only the mouth region, preserve everything else"

**Key difference:** StableAvatar's mask provides a strong spatial prior — the model only
needs to generate the mouth region. OmniAvatar's mask provides no spatial information;
the model must learn which regions to animate purely from the audio conditioning signal.

---

## 3. The Masked Video

### OmniAvatar: No Masked Video

OmniAvatar does **not** have a masked video concept. The conditioning `y` contains only:
- The first frame repeated (appearance reference)
- A binary mask (temporal indicator)

The model receives the noisy latent `x` and must generate the full frame conditioned on
the reference appearance + audio. There is no partial ground truth where some regions are
preserved and others need reconstruction.

### StableAvatar V2V: Mouth-Erased Video as Conditioning

StableAvatar explicitly feeds a **masked video** — the ground truth video with the mouth
region zeroed out — as part of the conditioning input.

**Pixel-space construction** (`precompute_vae_latents.py:144-146`):
```python
masked_video = video_tensor.clone()                 # (3, T, H, W) in [-1, 1]
mask_2d = mask.view(1, 1, H, W)                    # LatentSync mask: 1=keep, 0=mouth
masked_video[:, 1:, :, :] *= mask_2d               # Frame 0 untouched, frames 1+ masked
```

**Or in the dataset** (`talking_video_dataset_fantasy.py:450`):
```python
masked_pixel_values = tgt_pil_image_list * (1 - pixel_value_masks)
```

- Frame 0: complete original frame (identity anchor)
- Frames 1+: upper face preserved, mouth region set to black (0)

This masked video is then VAE-encoded to produce `masked_latents: (B, 16, 21, 64, 64)`,
which becomes channels 4–19 of the `y` tensor.

**Impact:** The model receives the upper face as "given" information and only needs to
generate the mouth region. This is a much easier task than OmniAvatar's approach where
the model must generate the entire face from scratch at every frame.

---

## 4. The Reference Frame(s)

### OmniAvatar: First Frame of Training Clip, Repeated

```python
ref_latent = input_latents[:, :, :1]    # (1, 16, 1, 64, 64) — first frame
y = ref_latent.repeat(1, 1, T_lat, 1, 1)  # repeated across all timesteps
```

- **Source:** Frame 0 of the training clip (same clip being denoised)
- **Format:** Single VAE-encoded frame, tiled identically across all latent timesteps
- **Purpose:** Provides appearance/identity information
- **Limitation:** Only one viewpoint; if the person moves significantly during the clip,
  the repeated reference doesn't capture the motion

### StableAvatar V2V: Separate Video Segment (81 frames, non-overlapping)

**Precomputation** (`precompute_ref_latents.py:74-78`):
```python
if total_frames >= 2 * num_frames:
    ref_start = num_frames          # non-overlapping: frames 81–161
else:
    ref_start = max(0, total_frames - num_frames)  # fallback: last available
```

- **Source:** A **different** 81-frame segment from the same video, non-overlapping with the
  training clip (training uses frames 0–80, reference uses frames 81–161)
- **Format:** Full 81-frame video segment, VAE-encoded to `(16, 21, 64, 64)` — preserves
  temporal variation, not just a single frozen frame
- **Purpose:** Provides rich identity and appearance information from multiple angles/expressions
- **Advantage:** The model sees the person from different moments in time, making identity
  preservation more robust

Additionally, StableAvatar feeds the reference frame through a **CLIP image encoder**
to produce `clip_context: (B, 257, 4096)`, which is concatenated with text embeddings
in the cross-attention context. OmniAvatar does not use CLIP at all.

---

## 5. Audio Processing

### OmniAvatar: Full Wav2Vec2 Hidden State Hierarchy (10752-dim)

```python
hidden_states = audio_encoder(input_values, seq_len=num_frames, output_hidden_states=True)
audio_emb = hidden_states.last_hidden_state               # (1, T, 768)
for hs in hidden_states.hidden_states:                     # 13 more tensors
    audio_emb = torch.cat((audio_emb, hs), dim=-1)
# Result: (1, T, 10752)  — 14 × 768
```

- Uses **all 14 hidden states** from Wav2Vec2 (last output + 13 intermediate including
  feature projection), concatenated along the feature dimension
- Injected via **AudioPack** (temporal patchify [4,1,1] + Linear(43008→32) + LayerNorm)
  followed by **19 separate Linear(32→5120) projections** (one per eligible layer)
- **Additive residual** at DiT layers 2–20 only (first 2 and second half excluded)
- CFG dropout: audio zeroed with configurable probability during training

### StableAvatar V2V: Wav2Vec2 Last Hidden State (768-dim) + Vocal Projector

```python
hidden_states = wav2vec(input_values)
vocal_embeddings = hidden_states.last_hidden_state   # (1, T_audio, 768)
```

- Uses **only the last hidden state** (768-dim), not all intermediate layers
- Processed by a **FantasyTalkingVocalCondition** projector module that maps
  `(T_audio, 768) → (T_lat, model_dim)` and interacts with the transformer blocks
- CFG dropout: 10% probability of zeroing audio embeddings

---

## 6. Model Input Channel Breakdown

### OmniAvatar (33 channels → patch_embedding)

```
x (noisy latent):           16ch  [noise]
y (reference + mask):       17ch  [16ch ref_frame_repeated + 1ch binary_mask]
─────────────────────────────────
Total into patch_embedding: 33ch
```

### StableAvatar V2V with ref (52 channels → patch_embedding)

```
x (noisy latent):           16ch  [noise]
y (mask + masked + ref):    36ch  [4ch spatial_mask + 16ch masked_video + 16ch ref_sequence]
─────────────────────────────────
Total into patch_embedding: 52ch
```

---

## 7. Loss Computation

### OmniAvatar

```python
loss = F.mse_loss(noise_pred.float(), training_target.float())
```

- Unweighted MSE over the entire latent volume
- Optional auxiliary losses (SyncNet, LPIPS, TREPA) when enabled
- No spatial weighting — loss is uniform across face, hair, background

### StableAvatar V2V

```python
# Randomly select loss weighting strategy:
r = random.random()
if 0.4 <= r < 0.5:
    loss = MSE(pred, target) * tgt_face_masks         # face-only loss
elif r >= 0.5:
    loss = MSE(pred, target) * tgt_lip_masks           # lip-only loss
else:
    loss = MSE(pred, target) * (1 + face + lip_masks)  # weighted everywhere

# Flow matching sigma weighting
loss *= compute_loss_weighting_for_sd3(sigmas)

# Optional motion sub-loss
if motion_sub_loss:
    gt_motion = target[:, 1:] - target[:, :-1]
    pred_motion = pred[:, 1:] - pred[:, :-1]
    loss = 0.75 * loss + 0.25 * MSE(pred_motion, gt_motion)
```

- **Spatially weighted** — randomly focuses on face region, lip region, or upweights both
- **Sigma-weighted** — flow matching loss weighting (sd3-style)
- **Motion sub-loss** — penalizes temporal inconsistency in inter-frame differences

---

## 8. Architectural Implications

### Why StableAvatar's approach is "easier" for the model

1. **Masked video as input**: The model receives the upper face as given information;
   it only needs to generate the mouth area. This drastically reduces what needs to be learned.

2. **Spatial mask as explicit prior**: The 4-channel mask explicitly tells the model which
   spatial regions to reconstruct. No ambiguity about what to animate.

3. **Reference from different segment**: 81 frames from a different part of the video provide
   richer identity information than a single repeated frame.

4. **CLIP features**: Additional semantic understanding of the reference appearance.

### Why OmniAvatar's approach is "harder" but potentially more flexible

1. **No masked video**: The model must generate the entire frame, learning from scratch which
   regions correspond to speech (mouth, jaw, cheeks) vs. static regions.

2. **No spatial mask**: The binary temporal mask gives zero spatial guidance. The audio
   conditioning signal must carry all the information about what to animate.

3. **Single reference frame**: Less identity information, but simpler pipeline.

4. **Richer audio features**: 10752-dim vs 768-dim audio features may compensate for the
   lack of explicit spatial guidance — the model can extract more nuanced speech information.

5. **Layer-selective injection**: AudioPack's bottleneck (43008→32→5120) and selective
   injection at layers 2–20 may help the model learn a cleaner separation between audio-driven
   and text/appearance-driven generation.

---

## 9. Key Files Reference

**OmniAvatar:**
- `scripts/train.py:366-371` — `prepare_reference_input()` (y construction)
- `scripts/train.py:528-576` — `forward()` (full training step)
- `OmniAvatar/models/wan_video_dit.py:350-403` — DiT audio injection + x,y concat
- `OmniAvatar/models/audio_pack.py` — AudioPack module

**StableAvatar:**
- `train_1B_square_v2v.sh` — Training shell script with all args
- `train_1B_square.py:1643-1663` — mask processing + y construction
- `train_1B_square.py:98-127` — `resize_mask()` function
- `train_1B_square.py:1240-1255` — Patch embedding expansion for ref channels
- `wan/dataset/talking_video_dataset_fantasy.py:192-197` — Precomputed mask creation
- `wan/dataset/talking_video_dataset_fantasy.py:440-456` — Live mask + masked video
- `precompute_vae_latents.py:144-146` — Masked video pixel-space construction
- `precompute_ref_latents.py:74-78` — Reference segment selection logic
