#!/bin/bash
# Run comprehensive evaluation: standard + GT-aligned metrics.
# Phase 1: all standard evals, Phase 2: all GT-aligned evals.
# Schedules across 4 GPUs, continues on individual job failure.
set -uo pipefail  # no -e: we handle errors per-job

cd /home/work/.local/OmniAvatar

BASE="demo_out/comprehensive_eval"
METRICS_REPO="/home/work/.local/latentsync-metrics"
CONDA_METRICS="/home/work/.local/miniconda3/envs/latentsync-metrics/bin"
CONDA_OMNI="/home/work/.local/miniconda3/envs/omniavatar/bin"
SHAPE_PRED="${METRICS_REPO}/shape_predictor_68_face_landmarks.dat"
PROJECT_DIR="/home/work/.local/OmniAvatar"

METHODS=(LatentSync Wav2Lip VideoReTalking Diff2Lip MuseTalk Maskall RefseqNew)
DATASETS=(hdtf hallo3)

FAILED=0

# ─── Standard evaluation (independent crops) ─────────────────────────────────

run_standard() {
    local gpu=$1 ds=$2 method=$3
    local real_dir="${PROJECT_DIR}/${BASE}/originals/${ds}"
    local fake_dir="${PROJECT_DIR}/${BASE}/methods/${ds}/${method}"
    local log_dir="${PROJECT_DIR}/${BASE}/metrics_standard/${ds}/${method}"
    mkdir -p "$log_dir"

    if [ -f "$log_dir/metrics.log" ] && grep -q "completed successfully" "$log_dir/metrics.log" 2>/dev/null; then
        echo "  [GPU $gpu] [SKIP] standard ${ds}/${method} (done)"
        return 0
    fi

    echo "  [GPU $gpu] standard ${ds}/${method}..."
    cd "$METRICS_REPO"
    if CUDA_VISIBLE_DEVICES=$gpu PATH="${CONDA_METRICS}:$PATH" \
        bash eval/run_metrics.sh \
        --real_videos_dir "$real_dir" \
        --fake_videos_dir "$fake_dir" \
        --shape_predictor_path "$SHAPE_PRED" \
        --output_dir "$log_dir" --log_path "$log_dir/metrics.log" \
        --fallback_detection_confidence 0.2 --all \
        > "$log_dir/eval.log" 2>&1; then
        cd "$PROJECT_DIR"
        echo "  [GPU $gpu] [DONE] standard ${ds}/${method}"
    else
        cd "$PROJECT_DIR"
        echo "  [GPU $gpu] [FAIL] standard ${ds}/${method} (see $log_dir/eval.log)"
        return 1
    fi
}

# GT self-eval (sync metrics only)
run_gt_sync() {
    local gpu=$1 ds=$2
    local real_dir="${PROJECT_DIR}/${BASE}/originals/${ds}"
    local log_dir="${PROJECT_DIR}/${BASE}/metrics_standard/${ds}/GT"
    mkdir -p "$log_dir"

    if [ -f "$log_dir/metrics.log" ] && grep -q "completed" "$log_dir/metrics.log" 2>/dev/null; then
        echo "  [GPU $gpu] [SKIP] GT sync ${ds} (done)"
        return 0
    fi

    echo "  [GPU $gpu] GT sync ${ds}..."
    cd "$METRICS_REPO"
    if CUDA_VISIBLE_DEVICES=$gpu PATH="${CONDA_METRICS}:$PATH" \
        bash eval/run_metrics.sh \
        --real_videos_dir "$real_dir" \
        --fake_videos_dir "$real_dir" \
        --shape_predictor_path "$SHAPE_PRED" \
        --output_dir "$log_dir" --log_path "$log_dir/metrics.log" \
        --fallback_detection_confidence 0.2 \
        --syncnet \
        > "$log_dir/eval.log" 2>&1; then
        cd "$PROJECT_DIR"
        echo "  [GPU $gpu] [DONE] GT sync ${ds}"
    else
        cd "$PROJECT_DIR"
        echo "  [GPU $gpu] [FAIL] GT sync ${ds} (see $log_dir/eval.log)"
        return 1
    fi
}

# ─── GT-aligned evaluation ────────────────────────────────────────────────────

run_aligned() {
    local gpu=$1 ds=$2 method=$3
    local real_dir="${PROJECT_DIR}/${BASE}/originals/${ds}"
    local fake_dir="${PROJECT_DIR}/${BASE}/methods/${ds}/${method}"
    local log_dir="${PROJECT_DIR}/${BASE}/metrics_gt_aligned/${ds}/${method}"
    mkdir -p "$log_dir"

    if [ -f "$log_dir/metrics_aligned.log" ]; then
        echo "  [GPU $gpu] [SKIP] aligned ${ds}/${method} (done)"
        return 0
    fi

    echo "  [GPU $gpu] aligned ${ds}/${method}..."
    if CUDA_VISIBLE_DEVICES=$gpu ${CONDA_METRICS}/python \
        "${PROJECT_DIR}/scripts/eval_aligned_crops.py" \
        --real_videos_dir "$real_dir" \
        --fake_videos_dir "$fake_dir" \
        --output_dir "$log_dir" \
        --device cuda:0 --metrics ssim fid fvd \
        > "$log_dir/eval.log" 2>&1; then
        echo "  [GPU $gpu] [DONE] aligned ${ds}/${method}"
    else
        echo "  [GPU $gpu] [FAIL] aligned ${ds}/${method} (see $log_dir/eval.log)"
        return 1
    fi
}

# ─── Helper: run a batch of jobs across 4 GPUs ──────────────────────────────

run_batch() {
    local -a batch_jobs=("$@")
    local gpu=0
    local -a pids=()

    for job in "${batch_jobs[@]}"; do
        IFS='|' read -r type ds method <<< "$job"
        case "$type" in
            standard)   run_standard $gpu "$ds" "$method" & ;;
            gt_sync)    run_gt_sync $gpu "$ds" & ;;
            aligned)    run_aligned $gpu "$ds" "$method" & ;;
        esac
        pids+=($!)
        gpu=$(( (gpu + 1) % 4 ))

        if [ ${#pids[@]} -ge 4 ]; then
            for pid in "${pids[@]}"; do
                wait "$pid" || FAILED=$((FAILED + 1))
            done
            pids=()
        fi
    done

    # Wait for remaining in this batch
    for pid in "${pids[@]}"; do
        wait "$pid" || FAILED=$((FAILED + 1))
    done
}

# ─── Phase 1: Standard evaluation (all methods × datasets) ──────────────────

echo "=== Phase 1: Standard Evaluation ==="
declare -a STANDARD_JOBS=()
for ds in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        STANDARD_JOBS+=("standard|${ds}|${method}")
    done
    STANDARD_JOBS+=("gt_sync|${ds}|GT")
done
echo "Standard jobs: ${#STANDARD_JOBS[@]}"
echo ""
run_batch "${STANDARD_JOBS[@]}"

echo ""
echo "=== Phase 1 Complete ==="
echo ""

# ─── Phase 2: GT-aligned evaluation (all methods × datasets) ────────────────

echo "=== Phase 2: GT-Aligned Evaluation ==="
declare -a ALIGNED_JOBS=()
for ds in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        ALIGNED_JOBS+=("aligned|${ds}|${method}")
    done
done
echo "Aligned jobs: ${#ALIGNED_JOBS[@]}"
echo ""
run_batch "${ALIGNED_JOBS[@]}"

echo ""
if [ $FAILED -gt 0 ]; then
    echo "=== EVALUATIONS COMPLETE ($FAILED jobs failed) ==="
else
    echo "=== ALL EVALUATIONS COMPLETE ==="
fi

# ─── Quick summary ────────────────────────────────────────────────────────────

echo ""
echo "=== Standard Metrics Summary ==="
for ds in "${DATASETS[@]}"; do
    echo "--- ${ds} ---"
    printf "%-16s %8s %8s %8s %8s %8s %8s %8s\n" "Method" "FID" "SSIM" "FVD" "CSIM" "Sync-C" "Sync-D" "LMD"
    for method in GT "${METHODS[@]}"; do
        dir="${BASE}/metrics_standard/${ds}/${method}"
        fid=$(grep "mean_fid:" "$dir/fid_per_video.log" 2>/dev/null | awk '{print $2}')
        ssim=$(grep "mean_ssim:" "$dir/ssim_lmd_per_video.log" 2>/dev/null | awk '{print $2}')
        lmd=$(grep "mean_lmd:" "$dir/ssim_lmd_per_video.log" 2>/dev/null | awk '{print $2}')
        fvd=$(grep "^FVD:" "$dir/metrics.log" 2>/dev/null | awk '{print $2}')
        csim=$(grep "^CSIM:" "$dir/metrics.log" 2>/dev/null | awk '{print $2}')
        syncc=$(grep "Mean SyncNet Confidence" "$dir/metrics.log" 2>/dev/null | tail -1 | awk '{print $NF}')
        syncd=$(grep "Mean SyncNet Min Distance" "$dir/metrics.log" 2>/dev/null | tail -1 | awk '{print $NF}')
        printf "%-16s %8s %8s %8s %8s %8s %8s %8s\n" \
            "$method" "${fid:--}" "${ssim:--}" "${fvd:--}" "${csim:--}" "${syncc:--}" "${syncd:--}" "${lmd:--}"
    done
    echo ""
done

echo "=== GT-Aligned Metrics Summary ==="
for ds in "${DATASETS[@]}"; do
    echo "--- ${ds} ---"
    printf "%-16s %8s %8s %8s\n" "Method" "FID" "SSIM" "FVD"
    for method in "${METHODS[@]}"; do
        dir="${BASE}/metrics_gt_aligned/${ds}/${method}"
        if [ -f "$dir/metrics_aligned.log" ]; then
            ssim=$(grep "^SSIM:" "$dir/metrics_aligned.log" | awk '{print $2}')
            fid=$(grep "^FID:" "$dir/metrics_aligned.log" | awk '{print $2}')
            fvd=$(grep "^FVD:" "$dir/metrics_aligned.log" | awk '{print $2}')
        else
            ssim="-"; fid="-"; fvd="-"
        fi
        printf "%-16s %8s %8s %8s\n" "$method" "${fid:--}" "${ssim:--}" "${fvd:--}"
    done
    echo ""
done
