# ODE Trajectory Analysis — Complete Experiment Inventory

## Common setup (all experiments)

- **Teacher model**: OmniAvatar 14B V2V (Wan 2.1 T2V-14B base + LoRA + audio modules)
  - Checkpoint: `/home/work/output_omniavatar_v2v_phase2/step-10500.pt`
  - Base weights: `pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-{00001..00006}-of-00006.safetensors`
  - `in_dim=65` (V2V + reference sequence)
- **Validation data**: 10 samples from `/home/work/stableavatar_data/v2v_validation_data/recon/`
  - Pre-aligned 512×512 face crops, 25fps, 250 frames per sample
  - Precomputed: `vae_latents_mask_all.pt`, `audio_emb_omniavatar.pt`, `text_emb.pt`, `ref_latents.pt`
- **ODE schedule**: 50 steps, shift=5.0 (matching OmniAvatar inference)
  - Timesteps: t ∈ [0.999, 0] with shift formula `t_shifted = shift·t / (1 + (shift-1)·t)`
- **Evaluation metrics** (per-step, computed on VAE-decoded 512×512 videos):
  - Reference (vs GT): pixel MSE, SSIM, LPIPS (AlexNet), LMD (dlib 68-point)
  - No-reference: sharpness (Laplacian variance), SyncNet Sync-C/Sync-D
  - Regions: mouth, upper_face, full (partitioned by LatentSync mask)
- **Evaluation script**: `scripts/eval_ode_perceptual_v2.py` (3 phases: decode, metrics, merge)
- **Spatial mask**: `/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png`
  - 1=upper_face (preserved), 0=mouth (generated). 41% mouth, 59% upper face.

## CFG drop modes

OmniAvatar conditions on **text (T5)** and **audio (Wav2Vec2)**. Two CFG schemes:

| Mode | `cfg_drop_text` | Unconditional branch | Use case |
|---|---|---|---|
| Text+audio (standard) | `true` | negative text + zero audio | Main paper §5 |
| Audio-only | `false` | positive text + zero audio | Supplementary; apples-to-apples with LatentSync |

noCFG (guidance_scale=1.0) runs only the conditional branch — drop mode is irrelevant.

## Experiment types

### 50-step trajectory
Full denoising from pure noise. At each step k, saves `step_k_xt.pt` (noisy state) and `step_k_x0.pt` (teacher's denoised prediction). This IS normal inference with intermediates saved.

**Script**: `FastGen/scripts/generate_omniavatar_ode_pairs_full.py`
**Key flags**: `--guidance_scale`, `--cfg_drop_text`, `--cfg_crossover` (optional scheduled CFG)

### Euler jump (2-step prediction)
NOT a full trajectory. At step 0, run teacher once to get a velocity estimate. Use that to JUMP directly to each timestep t_k in one shot. Re-evaluate teacher at the jumped x_t_k. Two CFG knobs:
- `cfg_step0`: drives the initial velocity (used for jumping)
- `cfg_teacher`: drives the re-evaluation at the jumped point

At step 0, the jump is a no-op (x_t_0_JUMPED = x_t_0), so the saved x0_pred uses cfg_teacher.

**Script**: `scripts/generate_single_step_predictions.py`
**Key flags**: `--mode euler_jump`, `--cfg_step0`, `--cfg_teacher`, `--cfg_drop_text`

### Fresh noise
At each timestep t_k, construct x_t in the training-distribution way:
`x_t = (1 − t_k) · x_0_GT + t_k · ε` (flow matching)
where x_0_GT is the ground-truth latent and ε is a fixed per-sample noise vector (same ε across all timesteps). Then run teacher once. Cleanest per-noise-level probe, free from sampling-history drift.

**Script**: `scripts/generate_single_step_predictions.py`
**Key flags**: `--mode fresh_noise`, `--guidance_scale`, `--cfg_drop_text`

### Scheduled CFG
50-step trajectory where CFG varies by step: steps 0 to τ-1 run at CFG=1.0 (noCFG), steps τ to 49 run at the specified guidance_scale. Tests whether the 2-step noCFG→CFG win comes from step count or CFG scheduling.

**Script**: `FastGen/scripts/generate_omniavatar_ode_pairs_full.py`
**Key flags**: `--cfg_crossover τ`, `--guidance_scale`, `--cfg_drop_text`

---

## Group 1: OmniAvatar 14B — Text+audio CFG (main paper §5)

**cfg_drop_text=true** (standard). Trajectory source: `/home/work/ode_full_trajectories/14B/`

| Experiment | Trajectory dir | Metrics dir | Rows | Status |
|---|---|---|---|---|
| 50-step CFG=4.5 | `ode_full_trajectories/14B/` | `ode_analysis/14B/perceptual_v2/` | 6031 | ✅ |
| 50-step noCFG | `ode_full_trajectories/14B/` ¹ | `ode_analysis/14B/trajectory_nocfg/` | 6031 | ✅ |
| Euler: CFG→CFG | — | `ode_analysis/14B/euler_cfg45_cfg45/` | 6031 | ✅ |
| Euler: noCFG→CFG | — | `ode_analysis/14B/euler_nocfg_cfg45/` | 6031 | ✅ |
| Euler: noCFG→noCFG | — | `ode_analysis/14B/euler_nocfg_nocfg/` | 6031 | ✅ |
| Euler: CFG→noCFG | — | `ode_analysis/14B/euler_cfg45_nocfg/` | 6031 | ✅ |
| Fresh noise (CFG=4.5) | — | `ode_analysis/14B/fresh_noise/` | 6031 | ✅ |
| Scheduled CFG τ=25 | `ode_full_trajectories/14B_schedule25/` | `ode_analysis/14B_schedule25/endpoint/` | 3391 | 🔄 running |

¹ noCFG trajectory generated separately at `ode_full_trajectories/14B_nocfg/`; metrics reference the main trajectory's schedule.

**Generation commands (text+audio CFG)**:
```bash
# 50-step trajectory
generate_omniavatar_ode_pairs_full.py --guidance_scale 4.5 --cfg_drop_text true

# Euler variants (all use traj_dir=/home/work/ode_full_trajectories/14B)
generate_single_step_predictions.py --mode euler_jump --cfg_step0 4.5 --cfg_teacher 4.5 --cfg_drop_text true  # CFG→CFG
generate_single_step_predictions.py --mode euler_jump --cfg_step0 1.0 --cfg_teacher 4.5 --cfg_drop_text true  # noCFG→CFG
generate_single_step_predictions.py --mode euler_jump --cfg_step0 1.0 --cfg_teacher 1.0 --cfg_drop_text true  # noCFG→noCFG
generate_single_step_predictions.py --mode euler_jump --cfg_step0 4.5 --cfg_teacher 1.0 --cfg_drop_text true  # CFG→noCFG
generate_single_step_predictions.py --mode fresh_noise --guidance_scale 4.5 --cfg_drop_text true              # fresh noise

# Scheduled CFG
generate_omniavatar_ode_pairs_full.py --guidance_scale 4.5 --cfg_crossover 25 --cfg_drop_text true
```

## Group 2: OmniAvatar 14B — Audio-only CFG (supplementary)

**cfg_drop_text=false**. Trajectory source: `/home/work/.local/ode_full_trajectories/14B_audio_only_cfg/`

| Experiment | Metrics dir | Rows | Status |
|---|---|---|---|
| 50-step CFG=4.5 | `ode_analysis/14B_audio_only_cfg/perceptual_v2/` | 6031 | ✅ |
| 50-step noCFG | reused from Group 1 | — | ✅ |
| Euler: CFG→CFG | `ode_analysis/14B_audio_only_cfg/euler_cfg45_cfg45/` | 6031 | ✅ |
| Euler: noCFG→CFG | `ode_analysis/14B_audio_only_cfg/euler_nocfg_cfg45/` | 6031 | ✅ |
| Euler: noCFG→noCFG | `ode_analysis/14B_audio_only_cfg/euler_nocfg_nocfg/` | 6031 | ✅ |
| Euler: CFG→noCFG | `ode_analysis/14B_audio_only_cfg/euler_cfg45_nocfg/` | 6031 | ✅ |
| Fresh noise (CFG=4.5) | `ode_analysis/14B_audio_only_cfg/fresh_noise/` | 6031 | ✅ |
| Scheduled CFG τ=25 | `ode_analysis/14B_audio_only_cfg_schedule25/` | 6031 | ✅ |

Same commands as Group 1 but with `--cfg_drop_text false`.

## Group 3: Fixed-CFG sweep — Text+audio (main paper §5.6)

**cfg_drop_text=true** (pre-flag code, text+audio by default).

| CFG | Trajectory dir | Metrics dir | Rows | Status |
|---|---|---|---|---|
| 1.0 | `ode_full_trajectories/14B_cfg1.0/` | `ode_analysis/14B_cfg1.0/perceptual_v2/` | 6031 | ✅ (prior session) |
| 3.0 | `ode_full_trajectories/14B_cfg3.0/` | `ode_analysis/14B_cfg3.0/perceptual_v2/` | 6031 | ✅ (prior session) |
| 4.5 | `ode_full_trajectories/14B/` | `ode_analysis/14B/perceptual_v2/` | 6031 | ✅ (same as Group 1) |
| 6.0 | `ode_full_trajectories/14B_cfg6.0/` | `ode_analysis/14B_cfg6.0/perceptual_v2/` | 6031 | ✅ (prior session) |

For the §5.6 Pareto plot, use the ENDPOINT (step 49) metrics from each.

## Group 4: Fixed-CFG sweep — Audio-only (supplementary)

**cfg_drop_text=false**.

| CFG | Trajectory dir | Metrics dir | Rows | Status |
|---|---|---|---|---|
| 1.0 | `ode_full_trajectories/14B_audio_only_cfg1.0/` | `ode_analysis/14B_audio_only_cfg1.0/endpoint/` | — | 🔄 queued |
| 3.0 | `ode_full_trajectories/14B_audio_only_cfg3.0/` | `ode_analysis/14B_audio_only_cfg3.0/endpoint/` | — | 🔄 queued |
| 4.5 | `ode_full_trajectories/14B_audio_only_cfg/` | `ode_analysis/14B_audio_only_cfg/perceptual_v2/` | 6031 | ✅ (same as Group 2) |
| 6.0 | `ode_full_trajectories/14B_audio_only_cfg6.0/` | `ode_analysis/14B_audio_only_cfg6.0/endpoint/` | — | 🔄 queued |

## Group 5: LatentSync 1.6 (512×512) (supplementary)

Separate model: UNet3D + DDIM scheduler + SD VAE (4ch) + Whisper tiny audio encoder.
CFG is natively audio-only (no text conditioning). Default CFG=1.5.
Analyses first 16 frames per sample (one DDIM chunk).
SyncNet uses `--sync_min_track 15` (lowered from 50 for 16-frame videos).

| Experiment | Metrics dir | Rows | Status |
|---|---|---|---|
| 50-step CFG=1.5 | `ode_analysis/latentsync_1.6/perceptual_v2/` | 5981 | ✅ |
| 50-step noCFG | `ode_analysis/latentsync_1.6/trajectory_nocfg/` | 5981 | ✅ |
| Euler: CFG→CFG | `ode_analysis/latentsync_1.6/euler_cfg15_cfg15/` | 5981 | ✅ |
| Euler: noCFG→CFG | `ode_analysis/latentsync_1.6/euler_nocfg_cfg15/` | 5981 | ✅ |
| Euler: noCFG→noCFG | `ode_analysis/latentsync_1.6/euler_nocfg_nocfg/` | 5981 | ✅ |
| Fresh noise (CFG=1.5) | `ode_analysis/latentsync_1.6/fresh_noise/` | 5981 | ✅ |

**Generation scripts** (LatentSync-specific, in `/home/work/.local/LatentSync/scripts/`):
- `generate_latentsync_ode_pairs_full.py` — 50-step DDIM trajectory
- `generate_latentsync_single_step.py` — Euler jump / fresh noise

Key difference from OmniAvatar: x0_pred derived from noise via DDIM closed form
`x0 = (x_t - sqrt(1-α̅)·ε̂) / sqrt(α̅)` rather than direct x0 prediction.

## Group 6: Spatial CFG-Difference Probe (Exp 2) (supplementary §S1.3)

At each (sample, step), feeds the SAME x_t to the 14B teacher twice (CFG=4.5 and CFG=1.0, audio-only), decodes both, computes |pred_CFG − pred_noCFG| per-pixel. Partitions by mouth mask.

| Protocol | Source of x_t | CSV | Rows | Status |
|---|---|---|---|---|
| fresh_noise | `(1-t)·GT + t·ε` | `spatial_cfg_probe/fresh_noise/spatial_cfg_probe.csv` | 4501 | ✅ |
| trajectory_cfg | CFG=4.5 trajectory's saved x_t | `spatial_cfg_probe/trajectory_cfg/spatial_cfg_probe.csv` | 4501 | ✅ |
| trajectory_nocfg | noCFG trajectory's saved x_t | `spatial_cfg_probe/trajectory_nocfg/spatial_cfg_probe.csv` | 4501 | ✅ |

Each protocol includes noise-floor control (x_t vs x_t+δ at σ=0.01).

**Script**: `scripts/spatial_cfg_probe.py`
**Output per protocol**: scalar CSV + `maps/{sample}/step_XXX_diff.npy` (512×512 diff maps)
**Metrics**: `cfg_diff_raw`, `cfg_diff_relative`, `noise_floor` per (step, sample, region)

## Other trajectories on disk (not in current scope)

| Dir | What | From |
|---|---|---|
| `1.3B/` | 1.3B model trajectory | Prior session |
| `14B_no_audio/` | Audio-ablation (zeroed audio) | Prior session |
| `videopainter_*/` | VideoPainter model | Prior session |
| `14B_cfg*.0/euler_perceptual*/` | Earlier Euler analysis format (different CSV structure) | Prior session |

---

## Pending evaluation (currently running on GPU 3)

1. **14B_schedule25 (text+audio) full metrics**: 🔄 ~56% done (~3391/6031 rows)
2. **14B_audio_only_cfg1.0 full metrics**: ⏳ queued
3. **14B_audio_only_cfg3.0 full metrics**: ⏳ queued
4. **14B_audio_only_cfg6.0 full metrics**: ⏳ queued

Each takes ~5-6h on 1 GPU. Sequential queue on GPU 3.

---

## Plot scripts (all in `/home/work/.local/OmniAvatar/scripts/`)

| Script | What it produces |
|---|---|
| `plot_combined_ode_comparison.py` | Original 14B text+audio: all variants overlaid |
| `plot_combined_ode_comparison_audio_only.py` | Audio-only 14B: all variants overlaid |
| `plot_combined_ode_comparison_latentsync.py` | LatentSync 1.6: all variants overlaid |
| `plot_trajectory_cfg_comparison.py` | CFG vs noCFG trajectory only |
| `plot_cfg_mode_compare.py` | Text+audio vs audio-only vs noCFG comparison |
| `plot_all_models_compare.py` | Cross-model (OmniAvatar vs LatentSync) overlay |
| `plot_exp1_schedule_compare.py` | Exp 1: scheduled CFG vs fixed CFG vs Euler variants |
| `plot_spatial_cfg_probe.py` | Exp 2: spatial probe line plots (mouth/upper ratio vs step) |
| `plot_spatial_cfg_heatmaps.py` | Exp 2: representative-timestep Δ heatmaps |

## Key code modifications made in this session

| File | Change | Commit |
|---|---|---|
| `FastGen/.../generate_omniavatar_ode_pairs_full.py` | Added `--cfg_drop_text`, `--cfg_crossover` flags | `7caa7a8`, `8347195` |
| `OmniAvatar/scripts/generate_single_step_predictions.py` | Added `--cfg_drop_text` flag | `18697dd` |
| `OmniAvatar/scripts/eval_ode_perceptual_v2.py` | Added `--vae_type {wan,sd}`, `--sync_min_track`, `--sync_only` | `01eb892`, `6c90753` |
| `OmniAvatar/scripts/spatial_cfg_probe.py` | New: spatial CFG-difference probe | `782fed2`, `2f9584b` |
| `LatentSync/scripts/generate_latentsync_ode_pairs_full.py` | New: 50-step DDIM trajectory extractor | (LatentSync repo) |
| `LatentSync/scripts/generate_latentsync_single_step.py` | New: Euler jump / fresh noise for LatentSync; fixed step-0 bug | (LatentSync repo) |
