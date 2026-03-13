# Comprehensive Baseline + Model Evaluation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run consistent evaluation (standard + GT-aligned) across 5 baselines, 2 of our models, and GT across 3 datasets, producing summary CSVs.

**Architecture:** Create a unified eval directory with normalized video naming. Run two eval modes: (1) standard independent-crop metrics via existing pipeline, (2) GT-aligned metrics via our `eval_aligned_crops.py`. All results saved to `demo_out/comprehensive_eval/`.

**Tech Stack:** latentsync-metrics eval pipeline, our `eval_aligned_crops.py`, MediaPipe, InceptionV3, I3D, ArcFace, SyncNet, dlib.

---

## Scope

**Datasets:** hdtf_short (33 samples), hallo3_short (30 samples).
**Skipped:** hallo3_cross — different sample set from our hallo3_mixed (30 vs 12, different IDs).
Our models don't have outputs for the baselines' 30 cross-identity samples. Deferred.

---

## Data Inventory

### Datasets & Sample Counts

| Dataset | GT Location | Samples | Our Models | Baselines |
|---|---|---|---|---|
| hdtf_short | `shared_hdtf_data/` | 33 | maskall-5500, refseq_new-3000 (composited) | 5 methods |
| hallo3_short | `shared_hallo3_data/` | 30 | maskall-5500, refseq_new-3000 (composited) | 5 methods |
| hallo3_cross | `shared_hallo3_data/` (same videos) | 30 | **NONE yet** | 5 methods |

### Baseline Video Naming Conventions

Each baseline has different naming. The `video_key.py` matching logic strips `_cfr25` and
uses delimiter-based matching, which handles most cases.

| Baseline | hdtf naming | hallo3 naming | Subdir? |
|---|---|---|---|
| LatentSync | `{name}_synced.mp4` | `{id}_synced.mp4` | `final/` |
| Wav2Lip | `{name}_cfr25_wav2lip.mp4` | `{id}_cfr25_wav2lip.mp4` | flat |
| VideoReTalking | `{name}_cfr25_retalking.mp4` | `{id}_cfr25_retalking.mp4` | flat |
| Diff2Lip | `audios.{name}_videos_cfr.{name}_cfr25.mp4` | same pattern | flat |
| MuseTalk | `{name}_cfr25_{name}.mp4` | `{id}_cfr25_{id}.mp4` | `v15/` (hdtf) |
| Our maskall | `{name}_cfr25.mp4` | `{id}_cfr25.mp4` | flat |
| Our refseq_new | `{name}_cfr25.mp4` | `{id}_cfr25.mp4` | flat |

### GT Directories (used as `--real_videos_dir`)

- hdtf: `/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data/` (sub_clip.mp4 in subdirs)
- hallo3: `/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data/` (sub_clip.mp4 in subdirs)
- hallo3_cross: same as hallo3 (same video identities, different audio not in GT)

### Originals (flat, for eval pipeline)

Already set up:
- `demo_out/v2v_eval_maskall/originals/{hdtf,hallo3}/` (symlinks to sub_clip.mp4)
- `demo_out/v2v_eval_refseq_new/originals/{hdtf,hallo3}/` (symlinks)

---

## File Structure

```
demo_out/comprehensive_eval/
├── originals/                    # GT videos (flat symlinks)
│   ├── hdtf/                     # 33 videos: {name}_cfr25.mp4 → sub_clip.mp4
│   └── hallo3/                   # 30 videos: {id}_cfr25.mp4 → sub_clip.mp4
├── methods/                      # All method outputs (flat symlinks)
│   ├── hdtf/
│   │   ├── LatentSync/           # 33 videos
│   │   ├── Wav2Lip/              # 31 videos
│   │   ├── VideoReTalking/       # 33 videos
│   │   ├── Diff2Lip/             # 33 videos
│   │   ├── MuseTalk/             # 33 videos
│   │   ├── Maskall/              # 33 videos
│   │   └── RefseqNew/            # 33 videos
│   └── hallo3/
│       ├── (same structure)
│       └── ...
├── metrics_standard/             # Standard eval (independent crops)
│   ├── hdtf/
│   │   ├── LatentSync/           # metrics.log, fid_per_video.log, etc.
│   │   ├── GT/                   # GT sync metrics
│   │   └── ...
│   └── hallo3/
├── metrics_gt_aligned/           # GT-aligned eval (same bbox)
│   ├── hdtf/
│   │   ├── LatentSync/
│   │   └── ...
│   └── hallo3/
├── summary_standard.csv          # Aggregated standard metrics
└── summary_gt_aligned.csv        # Aggregated GT-aligned metrics
```

---

## Tasks

### Task 1: Create unified directory structure and symlinks

**Files:**
- Create: `scripts/setup_comprehensive_eval.sh`

- [ ] **Step 1: Write the setup script**

Creates `demo_out/comprehensive_eval/` with:
- `originals/{hdtf,hallo3}/` — flat symlinks from GT subdirs
- `methods/{hdtf,hallo3}/{Method}/` — symlinks to baseline videos, handling naming

Key naming normalization: the `run_metrics.sh` pipeline uses `video_key.py` which matches
via delimiter-based search against known GT keys. So baseline videos don't need renaming —
just symlinking into method-specific dirs. The pipeline handles matching.

For MuseTalk hdtf: symlink from `v15/*.mp4`, not the parent dir.
For LatentSync: symlink from `final/*.mp4`.

- [ ] **Step 2: Run setup script, verify counts**

Expected: 7 methods × {33 hdtf, 30 hallo3} + 2 GT dirs.

- [ ] **Step 3: Commit**

---

### Task 2: Run standard evaluation (independent crops) for all methods

**Files:**
- Create: `scripts/run_comprehensive_eval.sh`

- [ ] **Step 1: Write the eval runner script**

For each method × dataset:
1. Run `run_metrics.sh --all` with real=originals, fake=methods/{ds}/{method}
2. Save to `metrics_standard/{ds}/{method}/`

Schedule across 4 GPUs. ~7 methods × 2 datasets = 14 eval runs.
Each takes ~5-10 min → ~40 min on 4 GPUs.

Also run GT vs GT for sync metrics:
- `run_metrics.sh --syncnet` with real=originals, fake=originals (self-eval)

- [ ] **Step 2: Run eval, monitor for errors**
- [ ] **Step 3: Verify all metrics.log files exist with expected content**
- [ ] **Step 4: Commit**

---

### Task 3: Run GT-aligned evaluation for all methods

**Files:**
- Modify: `scripts/eval_aligned_crops.py` (add CSIM support for completeness)

- [ ] **Step 1: Add batch mode to eval_aligned_crops.py**

Add a `--methods_dir` flag that processes all method subdirs in one run, or just
run it in a loop from the shell script.

- [ ] **Step 2: Run GT-aligned eval for all methods × datasets**

For each method × dataset:
1. Run `eval_aligned_crops.py --metrics ssim fid fvd`
2. Save to `metrics_gt_aligned/{ds}/{method}/`

Schedule across 4 GPUs. Same 14 runs.

- [ ] **Step 3: Verify results**
- [ ] **Step 4: Commit**

---

### Task 4: Aggregate results into summary CSVs

**Files:**
- Create: `scripts/aggregate_eval_results.py`

- [ ] **Step 1: Write aggregation script**

Reads all `metrics_standard/{ds}/{method}/` and `metrics_gt_aligned/{ds}/{method}/`
directories. Extracts metrics from `metrics.log`, `fid_per_video.log`,
`ssim_lmd_per_video.log`. Produces two CSVs:

```
summary_standard.csv:
method,dataset,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD

summary_gt_aligned.csv:
method,dataset,FID,SSIM,FVD
```

- [ ] **Step 2: Run aggregation, verify CSVs**
- [ ] **Step 3: Commit**

---

### Task 5 (Follow-up): hallo3_cross evaluation

**Deferred — requires preparing cross-identity inference inputs.**

- [ ] **Step 1: Prepare cross-identity input files for our models**

Need: video from one person + audio from another. Check if hallo3_cross
input files exist in `/tmp/v2v_eval_inputs/` or need creation.

- [ ] **Step 2: Run inference for maskall + refseq_new on hallo3_cross**
- [ ] **Step 3: Run standard + GT-aligned eval on hallo3_cross**
- [ ] **Step 4: Update summary CSVs**

---

## Verification Checklist

1. All method × dataset combinations have metrics
2. CSV column order matches existing: `method,dataset,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD`
3. GT sync metrics (Sync-C, Sync-D) are computed as baseline reference
4. Video counts match expectations (handle Wav2Lip missing samples)
5. Standard and GT-aligned FID/SSIM show expected pattern (GT-aligned should be better)
6. All results saved in `demo_out/comprehensive_eval/`

## Baseline Missing Videos

| Baseline | hdtf | hallo3 |
|---|---|---|
| Wav2Lip | 31/33 (2 missing) | 29/30 (1 missing) |
| VideoReTalking | 33/33 | 30/30 |
| Others | 33/33 | 30/30 |

The eval pipeline handles missing videos by computing metrics only on matched pairs.
