# ODE Analysis — OmniAvatar Audio-Only CFG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicate the existing OmniAvatar 14B ODE trajectory analysis with audio-only CFG (keep positive text embedding, zero only audio in the unconditional branch), producing an `ode_analysis/14B_audio_only_cfg/combined/` directory comparable to the original and a cross-CFG overlay plot.

**Architecture:** Add a single `--cfg_drop_text {true,false}` CLI flag to two existing generation scripts (FastGen's `generate_omniavatar_ode_pairs_full.py` and OmniAvatar's `generate_single_step_predictions.py`). When `false`, the negative condition keeps the positive text embedding. All other code (eval, metrics, plots) is reused as-is. The noCFG trajectory from the original run is reused (noCFG runs only the conditional branch).

**Tech Stack:** Python, PyTorch, existing FastGen / OmniAvatar codebases, conda envs `fastgen` + `omniavatar`, 4× H200 GPUs.

---

## File Structure

**Modify:**
- `/home/work/.local/hyunbin/FastGen/scripts/generate_omniavatar_ode_pairs_full.py` — add `--cfg_drop_text` flag and use it when building `neg_condition`.
- `/home/work/.local/OmniAvatar/scripts/generate_single_step_predictions.py` — mirror the flag.

**Create:**
- `/home/work/.local/OmniAvatar/scripts/plot_cfg_mode_compare.py` — new overlay plot comparing original-CFG, audio-only-CFG, and noCFG trajectories on the same axes.

**Outputs produced:**
- `/home/work/.local/ode_full_trajectories/14B_audio_only_cfg/` — trajectory latents
- `/home/work/.local/ode_analysis/14B_audio_only_cfg/{perceptual_v2,euler_cfg45_cfg45,euler_nocfg_cfg45,euler_nocfg_nocfg,fresh_noise,combined}/`
- `/home/work/.local/ode_analysis/14B_cfg_mode_compare/` — cross-CFG overlay plots

---

## Task 1: Add `--cfg_drop_text` flag to FastGen trajectory generator

**Files:**
- Modify: `/home/work/.local/hyunbin/FastGen/scripts/generate_omniavatar_ode_pairs_full.py`

**Context:** This script currently builds `neg_condition` in `load_sample()` (lines 220–231) using negative text embedding and zero audio. The flag lets us keep the positive text embedding while still zeroing audio.

- [ ] **Step 1: Add CLI argument**

Edit `parse_args()` around the "Ablation" section (currently near line 111–112):

```python
    # Ablation
    parser.add_argument("--zero_audio", action="store_true", default=False,
                        help="Zero out audio embeddings (for audio ablation analysis)")
    parser.add_argument("--cfg_drop_text", type=str, default="true",
                        choices=["true", "false"],
                        help="If true (default), negative branch uses negative text embedding. "
                             "If false, negative branch keeps positive text embedding (audio-only CFG).")
```

- [ ] **Step 2: Thread the flag into `load_sample`**

Change `load_sample()` signature and `neg_condition` construction. Find the current definition (around line 143):

```python
def load_sample(
    sample_dir: str,
    mask: torch.Tensor,
    neg_text_embeds: torch.Tensor,
    num_video_frames: int = 81,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Optional[Dict[str, torch.Tensor]]:
```

Add `cfg_drop_text: bool = True,` before `device`. Then inside, change the neg_condition dict (lines 220–231):

```python
        # Negative condition (for CFG)
        if cfg_drop_text:
            neg_text = neg_text_embeds.to(device=device, dtype=dtype)
            if neg_text.dim() == 2:
                neg_text = neg_text.unsqueeze(0)
        else:
            # audio-only CFG: keep positive text
            neg_text = condition["text_embeds"]
        neg_condition = {
            "text_embeds": neg_text,                                    # [1, 512, 4096]
            "audio_emb": torch.zeros_like(condition["audio_emb"]),      # [1, 81, 10752]
            "ref_latent": condition["ref_latent"],                      # same ref
            "mask": condition["mask"],                                  # same mask
            "masked_video": condition["masked_video"],                  # same masked video
            "ref_sequence": condition["ref_sequence"],                  # same ref_sequence
        }
```

- [ ] **Step 3: Pass the flag from `main()` into `load_sample()`**

Find the call site inside the sample loop (around line 517):

```python
        sample = load_sample(
            sample_dir=sample_dir,
            mask=mask,
            neg_text_embeds=neg_text_embeds,
            num_video_frames=args.num_video_frames,
            cfg_drop_text=(args.cfg_drop_text == "true"),
            device=device,
            dtype=dtype,
        )
```

- [ ] **Step 4: Log the flag at startup**

Inside `main()` after the "Loading OmniAvatar teacher" log (around line 406), add:

```python
    if global_rank == 0:
        cfg_mode = "text+audio (original)" if args.cfg_drop_text == "true" else "audio-only"
        logger.info(f"CFG drop mode: {cfg_mode}")
```

- [ ] **Step 5: Smoke-check the flag by loading a sample and inspecting neg_condition text_embeds**

Run the following sanity test from the command line with both flag values and compare output shapes and a checksum. This does NOT run the full trajectory — just validates the condition dict.

```bash
cd /home/work/.local/hyunbin/FastGen

PY=/home/work/.local/miniconda3/envs/fastgen/bin/python
$PY -c "
import sys, os, torch
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from generate_omniavatar_ode_pairs_full import load_sample, load_mask

mask = load_mask('/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png')
neg = torch.load('/home/work/stableavatar_data/neg_text_emb.pt', map_location='cpu', weights_only=False)
if isinstance(neg, dict):
    neg = next(v for v in neg.values() if isinstance(v, torch.Tensor))
neg = neg.to(torch.bfloat16)

sample_dir = '/home/work/stableavatar_data/v2v_validation_data/recon/' + sorted(os.listdir('/home/work/stableavatar_data/v2v_validation_data/recon'))[0]

s1 = load_sample(sample_dir, mask, neg, cfg_drop_text=True, device='cpu')
s2 = load_sample(sample_dir, mask, neg, cfg_drop_text=False, device='cpu')

t_pos = s1['condition']['text_embeds']
t_neg1 = s1['neg_condition']['text_embeds']
t_neg2 = s2['neg_condition']['text_embeds']

print('pos - neg1 L2:', (t_pos.float() - t_neg1.float()).norm().item())
print('pos - neg2 L2:', (t_pos.float() - t_neg2.float()).norm().item())
print('shapes:', t_pos.shape, t_neg1.shape, t_neg2.shape)
"
```

Expected: `pos - neg1 L2` is large (different embeddings), `pos - neg2 L2` is 0.0 (same tensor), shapes match.

- [ ] **Step 6: Commit**

```bash
cd /home/work/.local/hyunbin/FastGen
git add scripts/generate_omniavatar_ode_pairs_full.py
git commit -m "feat: add --cfg_drop_text flag for audio-only CFG ablation"
```

---

## Task 2: Generate audio-only-CFG 50-step trajectory (CFG=4.5)

**Files:** No code changes; execution only.

**GPU:** 1 GPU, ~60 minutes for 14B.

- [ ] **Step 1: Launch trajectory generation**

```bash
cd /home/work/.local/hyunbin/FastGen

PRETRAINED=/home/work/.local/OmniAvatar/pretrained_models
BASE_PATHS="${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,${PRETRAINED}/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors"
CKPT=/home/work/output_omniavatar_v2v_phase2/step-10500.pt
OUT=/home/work/.local/ode_full_trajectories/14B_audio_only_cfg
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 nohup /home/work/.local/miniconda3/envs/fastgen/bin/python \
    scripts/generate_omniavatar_ode_pairs_full.py \
    --model_size 14B --in_dim 65 \
    --base_model_paths "$BASE_PATHS" \
    --omniavatar_ckpt_path "$CKPT" \
    --data_dir /home/work/stableavatar_data/v2v_validation_data/recon \
    --latentsync_mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
    --neg_text_emb_path /home/work/stableavatar_data/neg_text_emb.pt \
    --output_dir "$OUT" \
    --num_inference_steps 50 --guidance_scale 4.5 --shift 5.0 \
    --max_samples 10 --skip_existing \
    --cfg_drop_text false \
    > "$OUT/trajectory.log" 2>&1 &

echo "PID=$!"
```

- [ ] **Step 2: Wait and verify**

Monitor:
```bash
tail -f /home/work/.local/ode_full_trajectories/14B_audio_only_cfg/trajectory.log
```

Expected completion log line: `Full ODE trajectory generation complete. Success: 10, Failed: 0, Total files: 1020, ...`

- [ ] **Step 3: Verify one sample's outputs**

```bash
ls /home/work/.local/ode_full_trajectories/14B_audio_only_cfg/ | head -5
FIRST=$(ls /home/work/.local/ode_full_trajectories/14B_audio_only_cfg | head -1)
ls /home/work/.local/ode_full_trajectories/14B_audio_only_cfg/$FIRST | wc -l
```

Expected: 10 sample directories, each with 102 files (50 xt + 50 x0 + ode_schedule.json + input_latents.pt).

---

## Task 3: Decode + metrics on audio-only-CFG trajectory

**Files:** No code changes; execution only. Runs the existing `eval_ode_perceptual_v2.py` unmodified.

**GPU:** 4 GPUs, ~2.5 hours.

- [ ] **Step 1: Phase 1 — decode (4 GPUs parallel)**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
TRAJ_DIR=/home/work/.local/ode_full_trajectories/14B_audio_only_cfg
OUTPUT_DIR=/home/work/.local/ode_analysis/14B_audio_only_cfg/perceptual_v2
mkdir -p "$OUTPUT_DIR"

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD $PY scripts/eval_ode_perceptual_v2.py \
        --phase decode \
        --traj_dir "$TRAJ_DIR" \
        --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
        --output_dir "$OUTPUT_DIR" \
        --shard_id $SHARD --num_shards 4 > "$OUTPUT_DIR/decode_shard${SHARD}.log" 2>&1 &
done
wait
echo "Decode done"
```

- [ ] **Step 2: Phase 2 — metrics (4 GPUs parallel)**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
TRAJ_DIR=/home/work/.local/ode_full_trajectories/14B_audio_only_cfg
OUTPUT_DIR=/home/work/.local/ode_analysis/14B_audio_only_cfg/perceptual_v2
MASK_PATH=/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png

for SHARD in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD $PY scripts/eval_ode_perceptual_v2.py \
        --phase metrics \
        --traj_dir "$TRAJ_DIR" \
        --mask_path "$MASK_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --shard_id $SHARD --num_shards 4 > "$OUTPUT_DIR/metrics_shard${SHARD}.log" 2>&1 &
done
wait
echo "Metrics done"
```

- [ ] **Step 3: Merge + plot**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
TRAJ_DIR=/home/work/.local/ode_full_trajectories/14B_audio_only_cfg
OUTPUT_DIR=/home/work/.local/ode_analysis/14B_audio_only_cfg/perceptual_v2

$PY scripts/eval_ode_perceptual_v2.py --merge \
    --traj_dir "$TRAJ_DIR" \
    --output_dir "$OUTPUT_DIR"
```

- [ ] **Step 4: Verify metrics.csv has 10 samples × 51 steps × 6 metrics**

```bash
head -3 /home/work/.local/ode_analysis/14B_audio_only_cfg/perceptual_v2/metrics.csv
wc -l /home/work/.local/ode_analysis/14B_audio_only_cfg/perceptual_v2/metrics.csv
```

Expected: header + N rows; N should be close to 10 samples × 51 step-indices × ~6 metrics × ~3 regions (pixel_mse, ssim, lpips, lmd, sharpness, sync_d, sync_c × mouth, upper_face, full, with only subsets per metric). Match the row count of the original `ode_analysis/14B/perceptual_v2/metrics.csv`.

---

## Task 4: Add `--cfg_drop_text` flag to single-step-predictions script

**Files:**
- Modify: `/home/work/.local/OmniAvatar/scripts/generate_single_step_predictions.py`

**Context:** This script runs Euler-jump and fresh-noise variants. `load_condition()` at line 70 constructs `neg_condition` at lines 113–117 using `neg_text_embeds`. Mirror the FastGen flag.

- [ ] **Step 1: Add the CLI argument**

Open `/home/work/.local/OmniAvatar/scripts/generate_single_step_predictions.py` and find the argument parser (search for `argparse.ArgumentParser` or `--mode`). Add after the `--cfg_teacher` or similar guidance argument:

```python
    parser.add_argument("--cfg_drop_text", type=str, default="true",
                        choices=["true", "false"],
                        help="If true (default), negative branch uses negative text embedding. "
                             "If false, negative branch keeps positive text embedding (audio-only CFG).")
```

- [ ] **Step 2: Extend `load_condition()` signature (line 70)**

Change:

```python
def load_condition(sample_name, mask, neg_text_embeds, device, dtype):
```

to:

```python
def load_condition(sample_name, mask, neg_text_embeds, device, dtype, cfg_drop_text=True):
```

- [ ] **Step 3: Branch on the flag at lines 113–117**

Current code at lines 113–117:

```python
    neg_text = neg_text_embeds.to(device=device, dtype=dtype)
    if neg_text.dim() == 2:
        neg_text = neg_text.unsqueeze(0)
    neg_condition = {
        "text_embeds": neg_text,
```

Replace with:

```python
    if cfg_drop_text:
        neg_text = neg_text_embeds.to(device=device, dtype=dtype)
        if neg_text.dim() == 2:
            neg_text = neg_text.unsqueeze(0)
    else:
        # audio-only CFG: keep positive text
        neg_text = condition["text_embeds"]
    neg_condition = {
        "text_embeds": neg_text,
```

- [ ] **Step 4: Thread flag into `load_condition()` call (line 241)**

Change:

```python
        condition, neg_condition, gt_latents_raw = load_condition(
            sample_name, mask, neg_text_embeds, device, dtype
        )
```

to:

```python
        condition, neg_condition, gt_latents_raw = load_condition(
            sample_name, mask, neg_text_embeds, device, dtype,
            cfg_drop_text=(args.cfg_drop_text == "true"),
        )
```

- [ ] **Step 5: Log flag at startup**

Near the top of `main()`, after args are parsed, add:

```python
    cfg_mode = "text+audio (original)" if args.cfg_drop_text == "true" else "audio-only"
    print(f"CFG drop mode: {cfg_mode}")
```

- [ ] **Step 6: Smoke test — compare neg_condition text with both flag values**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
$PY -c "
import sys, os, torch
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
# Import load_condition without running main
from generate_single_step_predictions import load_condition, NEG_TEXT_EMB
# Minimal setup
from PIL import Image
import numpy as np
import torch.nn.functional as F

mask_img = Image.open('/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png')
arr = np.array(mask_img, dtype=np.float32)
if arr.ndim == 3:
    arr = arr[:,:,0]
arr = arr / 255.0
t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
mask = F.interpolate(t, size=(64, 64), mode='bilinear', align_corners=False)
mask = (mask.squeeze() > 0.5).float()

neg = torch.load(NEG_TEXT_EMB, map_location='cpu', weights_only=False)
if isinstance(neg, dict):
    neg = next(v for v in neg.values() if isinstance(v, torch.Tensor))
neg = neg.to(torch.bfloat16)

name = sorted(os.listdir('/home/work/stableavatar_data/v2v_validation_data/recon'))[0]
c1, n1, _ = load_condition(name, mask, neg, 'cpu', torch.bfloat16, cfg_drop_text=True)
c2, n2, _ = load_condition(name, mask, neg, 'cpu', torch.bfloat16, cfg_drop_text=False)
print('drop_text=True  L2(neg, pos):', (n1[\"text_embeds\"].float() - c1[\"text_embeds\"].float()).norm().item())
print('drop_text=False L2(neg, pos):', (n2[\"text_embeds\"].float() - c2[\"text_embeds\"].float()).norm().item())
"
```

Expected: first value is large (>100); second is exactly 0.0.

- [ ] **Step 7: Commit**

```bash
cd /home/work/.local/OmniAvatar
git add scripts/generate_single_step_predictions.py
git commit -m "feat: --cfg_drop_text flag for audio-only CFG in single-step predictions"
```

---

## Task 5: Run Euler-jump + fresh-noise variants with audio-only CFG

**Files:** No code changes.

**GPU:** 4 GPUs, ~2 hours.

- [ ] **Step 1: Launch 4 variants in parallel**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
SCRIPT=scripts/generate_single_step_predictions.py
TRAJ_DIR=/home/work/.local/ode_full_trajectories/14B_audio_only_cfg
BASE_OUT=/home/work/.local/ode_analysis/14B_audio_only_cfg
mkdir -p "$BASE_OUT"

SAMPLES=$(ls -d ${TRAJ_DIR}/*/ | xargs -I{} basename {} | tr '\n' ',' | sed 's/,$//')

CUDA_VISIBLE_DEVICES=0 nohup $PY $SCRIPT \
    --mode fresh_noise --traj_dir "$TRAJ_DIR" \
    --output_dir "$BASE_OUT/fresh_noise" --samples "$SAMPLES" \
    --guidance_scale 4.5 --cfg_drop_text false --skip_existing \
    > "$BASE_OUT/fresh_noise.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup $PY $SCRIPT \
    --mode euler_jump --traj_dir "$TRAJ_DIR" \
    --output_dir "$BASE_OUT/euler_cfg45_cfg45" --samples "$SAMPLES" \
    --cfg_step0 4.5 --cfg_teacher 4.5 --cfg_drop_text false --skip_existing \
    > "$BASE_OUT/euler_cfg45_cfg45.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup $PY $SCRIPT \
    --mode euler_jump --traj_dir "$TRAJ_DIR" \
    --output_dir "$BASE_OUT/euler_nocfg_cfg45" --samples "$SAMPLES" \
    --cfg_step0 1.0 --cfg_teacher 4.5 --cfg_drop_text false --skip_existing \
    > "$BASE_OUT/euler_nocfg_cfg45.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup $PY $SCRIPT \
    --mode euler_jump --traj_dir "$TRAJ_DIR" \
    --output_dir "$BASE_OUT/euler_nocfg_nocfg" --samples "$SAMPLES" \
    --cfg_step0 1.0 --cfg_teacher 1.0 --cfg_drop_text false --skip_existing \
    > "$BASE_OUT/euler_nocfg_nocfg.log" 2>&1 &

wait
echo "All 4 variants done"
```

- [ ] **Step 2: Verify each variant produced decoded videos**

```bash
for V in fresh_noise euler_cfg45_cfg45 euler_nocfg_cfg45 euler_nocfg_nocfg; do
    N=$(find /home/work/.local/ode_analysis/14B_audio_only_cfg/$V/videos -name "step_*.mp4" 2>/dev/null | wc -l)
    echo "$V: $N videos"
done
```

Expected: each variant has ~500 videos (10 samples × 50 steps).

---

## Task 6: Run metrics on Euler + fresh noise variants

**Files:** No code changes.

**GPU:** 4 GPUs, 8–10 hours (SyncNet is the bottleneck).

- [ ] **Step 1: Launch metrics computation for all 4 variants**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
TRAJ_DIR=/home/work/.local/ode_full_trajectories/14B_audio_only_cfg
MASK_PATH=/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png
BASE_OUT=/home/work/.local/ode_analysis/14B_audio_only_cfg

GPU=0
for VARIANT in fresh_noise euler_cfg45_cfg45 euler_nocfg_cfg45 euler_nocfg_nocfg; do
    CUDA_VISIBLE_DEVICES=$GPU nohup $PY scripts/eval_ode_perceptual_v2.py \
        --phase metrics \
        --traj_dir "$TRAJ_DIR" \
        --mask_path "$MASK_PATH" \
        --output_dir "$BASE_OUT/$VARIANT" \
        > "$BASE_OUT/$VARIANT/metrics.log" 2>&1 &
    GPU=$(( (GPU + 1) % 4 ))
done
wait
echo "Metrics done"
```

- [ ] **Step 2: Merge + plot each variant**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
TRAJ_DIR=/home/work/.local/ode_full_trajectories/14B_audio_only_cfg
BASE_OUT=/home/work/.local/ode_analysis/14B_audio_only_cfg

for VARIANT in fresh_noise euler_cfg45_cfg45 euler_nocfg_cfg45 euler_nocfg_nocfg; do
    $PY scripts/eval_ode_perceptual_v2.py --merge \
        --traj_dir "$TRAJ_DIR" --output_dir "$BASE_OUT/$VARIANT"
done
```

- [ ] **Step 3: Verify each has metrics.csv**

```bash
for V in fresh_noise euler_cfg45_cfg45 euler_nocfg_cfg45 euler_nocfg_nocfg; do
    F=/home/work/.local/ode_analysis/14B_audio_only_cfg/$V/metrics.csv
    if [ -f "$F" ]; then
        echo "$V: $(wc -l < $F) rows"
    else
        echo "$V: MISSING"
    fi
done
```

---

## Task 7: Produce combined plots for audio-only CFG

**Files:** No code changes.

- [ ] **Step 1: Copy CSVs into combined directory**

```bash
BASE_OUT=/home/work/.local/ode_analysis/14B_audio_only_cfg
COMBINED=$BASE_OUT/combined
mkdir -p "$COMBINED"

cp $BASE_OUT/perceptual_v2/metrics.csv         $COMBINED/trajectory_cfg45.csv
cp /home/work/.local/ode_analysis/14B/trajectory_nocfg/metrics.csv $COMBINED/trajectory_nocfg.csv
cp $BASE_OUT/euler_cfg45_cfg45/metrics.csv     $COMBINED/euler_cfg45_cfg45.csv
cp $BASE_OUT/euler_nocfg_cfg45/metrics.csv     $COMBINED/euler_nocfg_cfg45.csv
cp $BASE_OUT/euler_nocfg_nocfg/metrics.csv     $COMBINED/euler_nocfg_nocfg.csv
cp $BASE_OUT/fresh_noise/metrics.csv           $COMBINED/fresh_noise.csv
```

Note: the noCFG trajectory reuses the original run's CSV (noCFG means only the conditional branch executes, so drop mode is irrelevant).

- [ ] **Step 2: Edit `plot_combined_ode_comparison.py` to point at new CSVs**

Duplicate the file as `plot_combined_ode_comparison_audio_only.py`:

```bash
cp /home/work/.local/OmniAvatar/scripts/plot_combined_ode_comparison.py \
   /home/work/.local/OmniAvatar/scripts/plot_combined_ode_comparison_audio_only.py
```

Then edit the `VARIANTS` dict paths (top of file) to point at the new `14B_audio_only_cfg/...` directories. The "trajectory" variant still uses the new audio-only CFG path; the "trajectory_nocfg" variant keeps the original `14B/trajectory_nocfg/metrics.csv`.

- [ ] **Step 3: Generate combined plots**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
$PY scripts/plot_combined_ode_comparison_audio_only.py \
    --output_dir /home/work/.local/ode_analysis/14B_audio_only_cfg/combined
```

Expected: `reference_metrics_combined.png` and `noref_metrics_combined.png` in the combined directory.

---

## Task 8: Create cross-CFG overlay plot (original vs audio-only vs noCFG)

**Files:**
- Create: `/home/work/.local/OmniAvatar/scripts/plot_cfg_mode_compare.py`

- [ ] **Step 1: Write the cross-CFG plot script**

```python
"""Plot 3-way comparison: original CFG vs audio-only CFG vs noCFG trajectory.

Usage:
    python scripts/plot_cfg_mode_compare.py \
        --output_dir /home/work/.local/ode_analysis/14B_cfg_mode_compare
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANTS = {
    "original_cfg": {
        "path": "/home/work/.local/ode_analysis/14B/perceptual_v2/metrics.csv",
        "label": "Original CFG (text+audio)",
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 2.5,
    },
    "audio_only_cfg": {
        "path": "/home/work/.local/ode_analysis/14B_audio_only_cfg/perceptual_v2/metrics.csv",
        "label": "Audio-only CFG",
        "color": "tab:red",
        "linestyle": "-",
        "linewidth": 2.5,
    },
    "nocfg": {
        "path": "/home/work/.local/ode_analysis/14B/trajectory_nocfg/metrics.csv",
        "label": "noCFG",
        "color": "black",
        "linestyle": "--",
        "linewidth": 2.0,
    },
}

TRAJ_DIR = "/home/work/.local/ode_full_trajectories/14B"


def load_schedule():
    samples = sorted([
        d for d in os.listdir(TRAJ_DIR)
        if os.path.isdir(os.path.join(TRAJ_DIR, d))
        and os.path.isfile(os.path.join(TRAJ_DIR, d, "ode_schedule.json"))
    ])
    with open(os.path.join(TRAJ_DIR, samples[0], "ode_schedule.json")) as f:
        schedule = json.load(f)
    return schedule["t_list"], schedule["num_steps"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    t_list, num_steps = load_schedule()
    t_values = np.array(t_list[:num_steps])

    dfs = {}
    gt_baselines = {}
    for name, cfg in VARIANTS.items():
        df = pd.read_csv(cfg["path"])
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        gt_agg = df[df["step"] == -1].groupby(["metric", "region"])["value"].mean()
        for (metric, region), val in gt_agg.items():
            gt_baselines.setdefault((metric, region), val)
        df_steps = df[df["step"] >= 0]
        dfs[name] = df_steps.groupby(["step", "metric", "region"])["value"].mean().reset_index()
        print(f"Loaded {name}: {len(df_steps)} step rows")

    # Reference metrics: 2x4 layout identical to plot_combined_ode_comparison.py
    ref_metrics = [
        ("pixel_mse", "Pixel MSE (mouth)", True),
        ("ssim", "SSIM (mouth)", False),
        ("lpips", "LPIPS (mouth)", False),
        ("lmd", "LMD (mouth)", False),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(28, 10))
    fig.suptitle("50-Step Trajectory — CFG Mode Comparison",
                 fontsize=16, fontweight="bold")

    for col, (metric_name, title, use_log) in enumerate(ref_metrics):
        ax = axes[0, col]
        for name, cfg in VARIANTS.items():
            sub = dfs[name]
            data = sub[(sub["metric"] == metric_name) & (sub["region"] == "mouth")].sort_values("step")
            if len(data) == 0:
                continue
            ax.plot(data["step"], data["value"],
                    color=cfg["color"], linestyle=cfg["linestyle"],
                    linewidth=cfg["linewidth"], label=cfg["label"],
                    marker=".", markersize=2)
        ax.set_xlabel("ODE Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"t={t_values[i]:.2f}" for i in tick_pos])
        ax2.set_xlabel("Timestep t", fontsize=8)

        ax_uf = axes[1, col]
        if metric_name != "lmd":
            for name, cfg in VARIANTS.items():
                sub = dfs[name]
                data = sub[(sub["metric"] == metric_name) & (sub["region"] == "upper_face")].sort_values("step")
                if len(data) == 0:
                    continue
                ax_uf.plot(data["step"], data["value"],
                           color=cfg["color"], linestyle=cfg["linestyle"],
                           linewidth=cfg["linewidth"], label=cfg["label"],
                           marker=".", markersize=2)
            ax_uf.set_ylabel(f"{title.split('(')[0]}(Upper Face)")
            ax_uf.set_title(f"{title.split('(')[0]}(Upper Face)")
        else:
            for name, cfg in VARIANTS.items():
                sub = dfs[name]
                data = sub[(sub["metric"] == "lmd") & (sub["region"] == "mouth")].sort_values("step")
                if len(data) == 0:
                    continue
                vals = data["value"].values
                delta = np.zeros(len(vals))
                delta[1:] = vals[:-1] - vals[1:]
                ax_uf.plot(data["step"].values, delta,
                           color=cfg["color"], linestyle=cfg["linestyle"],
                           linewidth=cfg["linewidth"], label=cfg["label"],
                           marker=".", markersize=2)
            ax_uf.axhline(y=0, color="gray", linewidth=0.5)
            ax_uf.set_ylabel("Δ LMD (improvement)")
            ax_uf.set_title("Per-Step Δ LMD")

        ax_uf.set_xlabel("ODE Step")
        ax_uf.legend(fontsize=9)
        ax_uf.grid(True, alpha=0.3)
        if use_log and metric_name != "lmd":
            ax_uf.set_yscale("log")

    plt.tight_layout()
    path1 = os.path.join(args.output_dir, "cfg_mode_compare_reference.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"Saved {path1}")

    # No-reference metrics
    noref_metrics = [
        ("sharpness", "Mouth Sharpness (Laplacian var)"),
        ("sync_d", "Sync-D (lower=better)"),
        ("sync_c", "Sync-C (higher=better)"),
    ]

    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))
    fig2.suptitle("50-Step Trajectory — CFG Mode Comparison (No-Reference)",
                  fontsize=14, fontweight="bold")

    for col, (metric_name, title) in enumerate(noref_metrics):
        ax = axes2[col]
        for name, cfg in VARIANTS.items():
            sub = dfs[name]
            data = sub[(sub["metric"] == metric_name) & (sub["region"] == "mouth")].sort_values("step")
            if len(data) == 0:
                continue
            ax.plot(data["step"], data["value"],
                    color=cfg["color"], linestyle=cfg["linestyle"],
                    linewidth=cfg["linewidth"], label=cfg["label"],
                    marker=".", markersize=2)

        gt_key = (metric_name, "mouth")
        if gt_key in gt_baselines:
            gt_val = gt_baselines[gt_key]
            ax.axhline(y=gt_val, color="green", linestyle="--",
                       linewidth=2, label=f"GT ({gt_val:.2f})")

        ax.set_xlabel("ODE Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"t={t_values[i]:.2f}" for i in tick_pos])
        ax2.set_xlabel("Timestep t", fontsize=8)

    plt.tight_layout()
    path2 = os.path.join(args.output_dir, "cfg_mode_compare_noref.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved {path2}")


if __name__ == "__main__":
    main()
```

Write this to `/home/work/.local/OmniAvatar/scripts/plot_cfg_mode_compare.py`.

- [ ] **Step 2: Generate the plot**

```bash
cd /home/work/.local/OmniAvatar
PY=/home/work/.local/miniconda3/envs/omniavatar/bin/python
mkdir -p /home/work/.local/ode_analysis/14B_cfg_mode_compare
$PY scripts/plot_cfg_mode_compare.py \
    --output_dir /home/work/.local/ode_analysis/14B_cfg_mode_compare
```

Expected: `cfg_mode_compare_reference.png` and `cfg_mode_compare_noref.png` in that directory.

- [ ] **Step 3: Commit**

```bash
cd /home/work/.local/OmniAvatar
git add scripts/plot_cfg_mode_compare.py scripts/plot_combined_ode_comparison_audio_only.py
git commit -m "feat: plots for audio-only CFG ablation"
```

---

## Final verification

- [ ] All expected outputs exist:

```bash
echo "Per-variant metrics:"
for V in perceptual_v2 euler_cfg45_cfg45 euler_nocfg_cfg45 euler_nocfg_nocfg fresh_noise; do
    F=/home/work/.local/ode_analysis/14B_audio_only_cfg/$V/metrics.csv
    [ -f "$F" ] && echo "  OK $V" || echo "  MISSING $V"
done

echo "Combined plots:"
ls /home/work/.local/ode_analysis/14B_audio_only_cfg/combined/*.png 2>/dev/null
ls /home/work/.local/ode_analysis/14B_cfg_mode_compare/*.png 2>/dev/null
```
