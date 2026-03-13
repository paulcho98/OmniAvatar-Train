#!/bin/bash
# Re-run metrics for hdtf and hallo3 with corrected originals (validation_* dirs).
# Uses GPUs 2 and 3 in parallel. GPU 2 = hdtf, GPU 3 = hallo3.
set -euo pipefail

METRICS_REPO="/home/work/.local/latentsync-metrics"
CONDA_METRICS="/home/work/.local/miniconda3/envs/latentsync-metrics/bin"
SHAPE_PRED="${METRICS_REPO}/shape_predictor_68_face_landmarks.dat"
PROJECT_DIR="/home/work/.local/OmniAvatar"
OUTPUT_BASE="${PROJECT_DIR}/demo_out/v2v_eval_ablation"

COMBOS=(2500_nocfg 2500_noffo 5500_nocfg 5500_noffo 5500_cfg)

cd "$METRICS_REPO"

run_all_combos() {
    local gpu=$1
    local ds=$2
    for label in "${COMBOS[@]}"; do
        local comp_dir="${OUTPUT_BASE}/${label}/${ds}_composited"
        local real_dir="${OUTPUT_BASE}/originals/${ds}"
        local log_dir="${OUTPUT_BASE}/metrics/${label}/${ds}"

        # Wipe old results completely
        rm -rf "$log_dir"
        mkdir -p "$log_dir"

        echo "[GPU $gpu] Running ${label}/${ds}..."
        CUDA_VISIBLE_DEVICES=$gpu PATH="${CONDA_METRICS}:$PATH" \
            bash eval/run_metrics.sh \
            --real_videos_dir "$real_dir" \
            --fake_videos_dir "$comp_dir" \
            --shape_predictor_path "$SHAPE_PRED" \
            --output_dir "$log_dir" --log_path "$log_dir/metrics.log" \
            --fallback_detection_confidence 0.2 --fake_videos_top_level --all \
            > "$log_dir/eval.log" 2>&1
        echo "[GPU $gpu] Done ${label}/${ds}"
    done
}

echo "=== Re-running metrics for hdtf (GPU 2) and hallo3 (GPU 3) ==="

run_all_combos 2 hdtf &
pid2=$!

run_all_combos 3 hallo3 &
pid3=$!

echo "  GPU 2 (PID $pid2): hdtf × 5 combos"
echo "  GPU 3 (PID $pid3): hallo3 × 5 combos"

wait $pid2 && echo "GPU 2 complete." || echo "WARNING: GPU 2 had failures"
wait $pid3 && echo "GPU 3 complete." || echo "WARNING: GPU 3 had failures"

echo ""
echo "=== Results ==="
printf "%-14s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Combo" "Dataset" "FVD" "FID" "CSIM" "SSIM" "LMD" "Sync-D" "Sync-C"
echo "------------------------------------------------------------------------------------"
for label in "${COMBOS[@]}"; do
    for ds in hdtf hallo3; do
        log="${OUTPUT_BASE}/metrics/${label}/${ds}/metrics.log"
        [ -f "$log" ] || continue
        fvd=$(grep -oP 'FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        fid=$(grep -oP 'mean_fid:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        csim=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        ssim=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        lmd=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        syncd=$(grep -oP 'Sync-D.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        syncc=$(grep -oP 'Sync-C.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        printf "%-14s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
            "$label" "$ds" "$fvd" "$fid" "$csim" "$ssim" "$lmd" "$syncd" "$syncc"
    done
done
echo "Done."
