# ODE Trajectory Analysis — Multi-Model Replication

## Goal

Replicate the existing ODE trajectory analysis (documented in
`/home/work/.local/ode_analysis/14B/combined/REPLICATION_GUIDE.md`) for two
additional settings, producing directly comparable per-timestep metric curves
across all three:

1. **OmniAvatar 14B — audio-only CFG** (same model, drop only audio for the
   unconditional branch, keep positive text).
2. **LatentSync 1.6 (512×512)** — a separate UNet-based lipsync diffusion
   model with its own VAE, scheduler, conditioning, and input format.

The evaluation code (metrics, plotting, GT baselines, mouth/upper-face split)
stays unchanged so the three analyses are apples-to-apples.

## Non-Goals

- No retraining. No changes to OmniAvatar or LatentSync model weights.
- No changes to the metrics themselves (pixel MSE, SSIM, LPIPS, LMD,
  sharpness, Sync-C/D). Same mask, same dlib predictor, same SyncNet model.
- No support for arbitrary chunk lengths in LatentSync — we pick one
  canonical chunk per sample.

## Sub-project decomposition

The work splits cleanly into two sub-projects. Each gets its own
implementation plan after this spec is approved.

### Sub-project A — OmniAvatar audio-only CFG

Trivial, ~half day, reuses all existing scripts with one new CLI flag.

**Change surface**:
- `/home/work/.local/hyunbin/FastGen/scripts/generate_omniavatar_ode_pairs_full.py`
  — currently builds `neg_condition` with negative text embedding and zero
  audio (`load_sample`, lines 220–231). Add `--cfg_drop_text {true,false}`
  (default `true` = current behaviour). When `false`, `neg_condition["text_embeds"]`
  is the positive text embedding; audio stays zeroed.
- `/home/work/.local/OmniAvatar/scripts/generate_single_step_predictions.py`
  — mirror the same flag in the Euler-jump variants (cfg_step0, cfg_teacher
  still control guidance scale; the new flag controls what's dropped in the
  unconditional branch).

**Trajectories to generate**:
- `14B_audio_only_cfg` CFG=4.5 trajectory (audio-only CFG)
- `14B` nocfg trajectory — already exists, reuse (noCFG means only the
  conditional branch runs, so "what you'd drop" doesn't matter).
- Euler-jump variants under `ode_analysis/14B_audio_only_cfg/` with the new
  flag set.

**Deliverable**: combined plots under
`ode_analysis/14B_audio_only_cfg/combined/` plus a cross-CFG overlay plot at
`ode_analysis/14B_cfg_mode_compare/` comparing original-CFG, audio-only-CFG,
and noCFG on the same axes.

### Sub-project B — LatentSync 1.6 replication

Extensive, ~2–3 days. New generation scripts on the LatentSync side; minimal
VAE-abstraction change to the shared eval script.

**New scripts** (all under `/home/work/.local/LatentSync/scripts/`):
- `generate_latentsync_ode_pairs_full.py` — mirror of FastGen's
  `generate_omniavatar_ode_pairs_full.py`. Runs 50-step DDIM on one chunk per
  sample, saves `step_XXX_xt.pt`, `step_XXX_x0.pt`, `ode_schedule.json`,
  `input_latents.pt`.
- `generate_latentsync_single_step.py` — mirror of
  `generate_single_step_predictions.py`. Supports `--mode {fresh_noise,
  euler_jump}` and `--cfg_step0 / --cfg_teacher`.

**Modified shared script**:
- `/home/work/.local/OmniAvatar/scripts/eval_ode_perceptual_v2.py` — add
  `--vae_type {wan,sd}` flag. `sd` branch: load SD VAE via
  `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")`, undo
  LatentSync's scaling (`latents / scaling_factor + shift_factor`), decode to
  `[f c h w]` 2D output and permute to the expected `[T H W 3]` uint8 numpy
  layout. Phase 2 (metrics) is VAE-agnostic and needs no changes.

## Key design decisions

### 1. One 16-frame chunk per sample
LatentSync processes 16 frames per DDIM call; OmniAvatar does 81 in one pass.
To compare per-timestep behaviour cleanly, we analyse **one canonical chunk
per sample: the first 16 frames**. These frames correspond to frames 0–15 of
OmniAvatar's 81-frame output. The per-timestep curves are thus
time-aligned across models at a fixed temporal window.

Consequence: LatentSync metrics average over 16 frames vs OmniAvatar's 81.
Sync-C/D use a 16-frame window inside either case, so that's unaffected.
Reference metrics (MSE/SSIM/LPIPS/LMD) will be noisier for LatentSync due to
smaller sample size, but the trajectory shape is what we care about.

### 2. Save latents, decode later
Mirror OmniAvatar's separation: generation scripts save raw latent tensors,
the eval script decodes via VAE in a separate phase. Keeps the pipeline
model-agnostic at the metrics stage.

For LatentSync, `x0_pred` is derived from the predicted noise via
DDIM's closed form:

```
x0_pred = (x_t - sqrt(1 - α̅_t) · ε̂) / sqrt(α̅_t)
```

Using `scheduler.alphas_cumprod[t]`. This matches what DDIM's own
`scheduler.step()` computes internally.

### 3. VAE abstraction via `--vae_type` flag
Only the decode step differs:
- **Wan VAE** (OmniAvatar): `[16, 21, 64, 64]` → `[T, H, W, 3]` via
  `pipe.vae.model.decode(latents, scale)`.
- **SD VAE** (LatentSync): `[1, 4, 16, 64, 64]` → rearrange `b c f h w → (b f) c h w`,
  decode, apply `(latents / 0.18215) + 0` inverse, output to `[T, H, W, 3]`.

### 4. Euler-jump and fresh-noise variants for LatentSync
Direct mapping:
- **Euler-jump at step k**: load saved `step_k_xt.pt`, run UNet once with
  CFG=`cfg_step0`, get noise_pred, convert to x0_pred via DDIM formula.
- **Fresh-noise at step k**: sample fresh noise `ε ~ N(0, I)` with fixed
  seed, compute `x_t = sqrt(α̅_t) · x_0 + sqrt(1-α̅_t) · ε` using GT
  `input_latents.pt`, then single UNet pass.

CFG values for LatentSync: `{1.0, 1.5}` (default is 1.5, not 4.5).

### 5. Same 10 validation samples, same metrics
All three models run on
`/home/work/stableavatar_data/v2v_validation_data/recon/*/` — 10 samples.
LatentSync consumes `sub_clip.mp4` + `audio.wav` directly (doesn't need
precomputed text/audio/VAE embeddings). Same mouth mask, same SyncNet,
same dlib predictor.

## Output layout

```
ode_analysis/
├── 14B/                                    # existing (original text+audio CFG)
│   └── combined/                           # unchanged
├── 14B_audio_only_cfg/                     # sub-project A (NEW)
│   ├── perceptual_v2/                      # CFG trajectory
│   ├── euler_cfg45_cfg45/
│   ├── euler_nocfg_cfg45/
│   ├── euler_nocfg_nocfg/
│   ├── fresh_noise/
│   └── combined/
├── 14B_cfg_mode_compare/                   # cross-CFG overlay plots (NEW)
├── latentsync_1.6/                         # sub-project B (NEW)
│   ├── perceptual_v2/                      # CFG=1.5 trajectory
│   ├── trajectory_nocfg/                   # CFG=1.0 trajectory
│   ├── euler_cfg15_cfg15/
│   ├── euler_nocfg_cfg15/
│   ├── euler_nocfg_nocfg/
│   ├── fresh_noise/
│   └── combined/
└── all_models_combined/                    # cross-model overlay (NEW)
```

## Data flow

### OmniAvatar audio-only CFG (per sample)
```
{vae_latents, audio_emb, text_emb, ref_latents}  (precomputed)
         │
         ▼
 generate_omniavatar_ode_pairs_full.py  --cfg_drop_text false
         │
         ▼
 step_XXX_xt.pt, step_XXX_x0.pt, input_latents.pt, ode_schedule.json
         │
         ▼
 eval_ode_perceptual_v2.py  --vae_type wan  --phase decode
         │
         ▼
 videos/{sample}/step_XXX.mp4 + gt.mp4
         │
         ▼
 eval_ode_perceptual_v2.py  --phase metrics --merge
```

### LatentSync (per sample)
```
sub_clip.mp4 + audio.wav
         │
         ▼
 affine-align face, whisper encode, SD-VAE encode masked/ref
         │
         ▼
 generate_latentsync_ode_pairs_full.py  (one chunk: frames 0..15)
         │
         ▼
 step_XXX_xt.pt, step_XXX_x0.pt, input_latents.pt, ode_schedule.json
 (noise-space x_t, x0 derived from noise_pred via DDIM formula)
         │
         ▼
 eval_ode_perceptual_v2.py  --vae_type sd  --phase decode
         │
         ▼
 videos/{sample}/step_XXX.mp4 + gt.mp4   (16-frame clips, 512×512)
         │
         ▼
 same metrics pipeline as OmniAvatar
```

## Known risks and smoke tests

1. **SD VAE normalization wrong silently** — decode `input_latents.pt` and
   compare pixel MSE to decoded GT video. Should be ≈ 0.
2. **Timestep alignment** — DDIM timesteps are integer [0, 999] descending;
   flow-matching t is float [0, 1] descending. Plots use **step index
   (0..49)** on primary x-axis with each scheduler's `t` on a twin axis.
   This is already how the combined plots work.
3. **Chunk length mismatch** — metrics over 16 frames vs 81 frames. Document
   explicitly; Sync-C/D use 16-frame windows inside the video regardless.
4. **DDIM `scale_model_input` is a no-op** for DDIM but is called in the
   pipeline; matching it in the extracted trajectory avoids subtle drift.
5. **Face detection determinism** — LatentSync's affine_transform uses
   face detection per frame. Cache face detection results per sample so
   re-runs stay deterministic (`--face_cache_dir` already exists in
   OmniAvatar's inference_v2v.py; add similar caching here).
6. **Audio padding** — LatentSync's whisper chunks may be shorter than 16 if
   the audio ends. For the recon samples (sub_clip.mp4 is already 81 frames
   = 3.24 s), 16 frames is ~0.64 s, well within audio length.

## Success criteria

- [ ] Sub-project A: `combined/reference_metrics_combined.png` and
      `combined/noref_metrics_combined.png` produced for
      `ode_analysis/14B_audio_only_cfg/combined/`. Plots look similar in
      structure to the original but reveal different CFG behaviour.
- [ ] Sub-project A: `ode_analysis/14B_cfg_mode_compare/` overlay plot with
      three trajectory variants (original, audio-only, noCFG) on same axes.
- [ ] Sub-project B: same artefact set under
      `ode_analysis/latentsync_1.6/combined/`.
- [ ] Cross-model plot `ode_analysis/all_models_combined/` showing
      OmniAvatar original CFG, OmniAvatar audio-only CFG, and LatentSync on
      the same axes for each metric. Step index on x-axis;
      both schedulers' t values labelled.
- [ ] Smoke test: LatentSync decode of `input_latents.pt` matches the
      aligned GT video within pixel MSE < 1e-3.
- [ ] All scripts accept the same `--skip_existing` and sharding flags as
      the existing ones.

## Open questions deferred to the plan

- Exact filename layout under `ode_analysis/latentsync_1.6/` — follow
  OmniAvatar's pattern exactly.
- Whether the LatentSync UNet forward pass needs any
  `scale_model_input` adjustment (check by comparing first DDIM step vs
  pipeline output).
- Whether to cache whisper features per sample (they're cheap to recompute;
  probably not worth it).
