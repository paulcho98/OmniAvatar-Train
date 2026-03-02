#!/bin/bash
# Composite V2V eval videos back onto originals and run full metrics.
#
# Phase 1: Composite all step/dataset combos (uses pre-existing face caches)
# Phase 2: Run latentsync-metrics for each combo
# Phase 3: Print summary table
#
# Usage: bash scripts/composite_and_eval.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ─── Configuration ───────────────────────────────────────────────────────────
STEPS="5000 4500 2500"
EVAL_BASE="demo_out/v2v_eval"
METRICS_REPO="/home/work/.local/latentsync-metrics"
SHAPE_PRED="/home/work/.local/latentsync-metrics/shape_predictor_68_face_landmarks.dat"
PYTHON="/home/work/.local/miniconda3/envs/omniavatar/bin/python"
GPUS=(0 1)  # Available GPUs

# Dataset config: name|originals_dir|gt_dir|existing_face_cache|metrics_flags
# gt_dir = ground truth for comparison (same as originals for recon datasets)
# existing_face_cache = pre-computed face caches from StableAvatar eval
DATASETS=(
    "hdtf|/home/work/.local/HDTF/HDTF_original_testset_81frames/videos_cfr|/home/work/.local/HDTF/HDTF_original_testset_81frames/videos_cfr|/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data/face_cache|--all"
    "hallo3|/home/work/.local/Hallo3_validation/validation_81frames/videos_cfr|/home/work/.local/Hallo3_validation/validation_81frames/videos_cfr|/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data/face_cache|--all"
)
# hallo3_mixed skipped — not face-aligned, compositing not applicable

# ─── Phase 1: Compositing ───────────────────────────────────────────────────
echo "=== Phase 1: Compositing ==="
for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name orig_dir gt_dir existing_cache metrics_flags <<< "$entry"
    FACE_CACHE="${EVAL_BASE}/face_cache/${name}"

    for step in $STEPS; do
        gen_dir="${EVAL_BASE}/step-${step}/${name}"
        comp_dir="${EVAL_BASE}/step-${step}/${name}_composited"

        # Skip if already done
        if [ -d "$comp_dir" ] && [ "$(ls "$comp_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  [SKIP] step-${step}/${name}_composited already exists"
            continue
        fi

        echo "  [RUN] step-${step}/${name} → ${comp_dir}"
        CUDA_VISIBLE_DEVICES=${GPUS[0]} $PYTHON scripts/composite_v2v.py \
            --generated_dir "$gen_dir" \
            --original_videos_dir "$orig_dir" \
            --output_dir "$comp_dir" \
            --face_cache_dir "$FACE_CACHE" \
            --existing_face_cache_dir "$existing_cache" \
            --num_frames 81 --fps 25
    done
done

# ─── Phase 2: Metrics ───────────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Metrics ==="
METRICS_LOG_BASE="${EVAL_BASE}/metrics"

gpu_idx=0
pids=()

for step in $STEPS; do
    for entry in "${DATASETS[@]}"; do
        IFS='|' read -r name orig_dir gt_dir existing_cache metrics_flags <<< "$entry"
        comp_dir="${EVAL_BASE}/step-${step}/${name}_composited"
        log_dir="${METRICS_LOG_BASE}/step-${step}/${name}"
        mkdir -p "$log_dir"

        # Skip if metrics already computed
        if [ -f "$log_dir/metrics.log" ]; then
            echo "  [SKIP] step-${step}/${name} metrics exist"
            continue
        fi

        gpu=${GPUS[$gpu_idx]}
        echo "  [GPU $gpu] step-${step}/${name}"

        CUDA_VISIBLE_DEVICES=$gpu bash "$METRICS_REPO/eval/run_metrics.sh" \
            --real_videos_dir "$gt_dir" \
            --fake_videos_dir "$comp_dir" \
            --shape_predictor_path "$SHAPE_PRED" \
            --output_dir "$log_dir" \
            --log_path "$log_dir/metrics.log" \
            --fallback_detection_confidence 0.2 \
            --fake_videos_top_level \
            $metrics_flags \
            > "$log_dir/eval.log" 2>&1 &
        pids+=($!)

        gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))

        # Wait when all GPUs busy
        if [ ${#pids[@]} -ge ${#GPUS[@]} ]; then
            for pid in "${pids[@]}"; do
                wait "$pid" || echo "WARNING: PID $pid failed"
            done
            pids=()
        fi
    done
done

# Wait for remaining
for pid in "${pids[@]}"; do
    wait "$pid" || echo "WARNING: PID $pid failed"
done

# ─── Phase 3: Summary ───────────────────────────────────────────────────────
echo ""
echo "=== Metrics Summary ==="
echo ""
printf "%-12s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Step" "Dataset" "FVD" "FID" "CSIM" "SSIM" "LMD" "Sync-D" "Sync-C"
echo "--------------------------------------------------------------------------------"

for step in $STEPS; do
    for name in hdtf hallo3; do
        log="${METRICS_LOG_BASE}/step-${step}/${name}/metrics.log"
        [ -f "$log" ] || continue

        fvd=$(grep -oP 'FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        fid=$(grep -oP 'mean_fid:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        csim=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        ssim=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        lmd=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        syncd=$(grep -oP 'Sync-D.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        syncc=$(grep -oP 'Sync-C.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")

        printf "%-12s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
            "step-${step}" "$name" "$fvd" "$fid" "$csim" "$ssim" "$lmd" "$syncd" "$syncc"
    done
done

echo ""
echo "Full logs: ${METRICS_LOG_BASE}/step-{step}/{dataset}/metrics.log"
