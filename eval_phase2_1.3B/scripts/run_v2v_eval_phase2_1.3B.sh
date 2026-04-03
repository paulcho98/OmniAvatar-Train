#!/bin/bash
# Phase 2 eval (1.3B): pipelined inference → composited metrics → GT-aligned crop metrics
# GPU 2 → hdtf, GPU 3 → hallo3 (parallel pipelines)
# 1.3B variant of run_v2v_eval_phase2.sh
#
# Usage: bash scripts/run_v2v_eval_phase2_1.3B.sh [STEP]
#   Default STEP=2500

set -euo pipefail

STEP="${1:-2500}"

# ─── Paths (all relative to PROJECT_DIR) ────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EXP_PATH="${PROJECT_DIR}/checkpoints"
INPUT_DIR="${PROJECT_DIR}/inputs"
OUTPUT_BASE="demo_out/v2v_eval_phase2_1.3B"
NUM_STEPS=50
GUIDANCE_SCALE=4.5

# Conda env paths — update these for your server
TORCHRUN="$(command -v torchrun || echo torchrun)"
METRICS_PYTHON="$(command -v python || echo python)"
CONDA_METRICS=""  # set to metrics env bin dir if different from current env

# Metrics repo (bundled in this package)
METRICS_REPO="${PROJECT_DIR}/latentsync-metrics"
SHAPE_PRED="${METRICS_REPO}/shape_predictor_68_face_landmarks.dat"

LABEL="step-${STEP}"

# Validation data (relative to project)
VAL_HDTF="${PROJECT_DIR}/validation_data/hdtf"
VAL_HALLO3="${PROJECT_DIR}/validation_data/hallo3"

# Face caches (inside validation data dirs)
FC_HDTF="${VAL_HDTF}/face_cache"
FC_HALLO3="${VAL_HALLO3}/face_cache"

# GT originals (same as validation data dirs)
GT_HDTF="$VAL_HDTF"
GT_HALLO3="$VAL_HALLO3"

# ─── Verify ──────────────────────────────────────────────────────────────────
echo "=== Phase 2 Eval (1.3B): ${LABEL} ==="
CKPT="${EXP_PATH}/${LABEL}.pt"
[ -f "$CKPT" ] || { echo "ERROR: $CKPT not found"; exit 1; }
echo "  Checkpoint: $CKPT"

# Generate input files if they don't exist
if [ ! -f "${INPUT_DIR}/hdtf_latentsync.txt" ] || [ ! -f "${INPUT_DIR}/hallo3_latentsync.txt" ]; then
    echo "  Input files not found, generating..."
    bash scripts/generate_input_files.sh
fi

for ds in hdtf hallo3; do
    [ -f "${INPUT_DIR}/${ds}_latentsync.txt" ] || { echo "ERROR: input file missing for ${ds}"; exit 1; }
done
echo "  Input files OK"

# ─── Setup originals ─────────────────────────────────────────────────────────
echo ""
echo "=== Setting up originals ==="

setup_originals() {
    local ds=$1 gt_dir=$2 out_dir=$3
    mkdir -p "$out_dir"
    if [ "$(ls "$out_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] ${ds} originals exist"
        return
    fi
    for d in "$gt_dir"/*/; do
        [ -d "$d" ] || continue
        local name=$(basename "$d")
        [ "$name" = "face_cache" ] && continue
        [ -f "$d/sub_clip.mp4" ] && ln -sf "$d/sub_clip.mp4" "$out_dir/${name}.mp4"
    done
    echo "  ${ds}: $(ls "$out_dir"/*.mp4 | wc -l) linked"
}

setup_originals hdtf "$GT_HDTF" "${OUTPUT_BASE}/originals/hdtf"
setup_originals hallo3 "$GT_HALLO3" "${OUTPUT_BASE}/originals/hallo3"

# ─── Helper functions ────────────────────────────────────────────────────────

run_inference() {
    local gpu=$1 ds=$2 face_cache=$3
    local out_dir="${OUTPUT_BASE}/${LABEL}/${ds}"
    local input_file="${INPUT_DIR}/${ds}_latentsync.txt"
    local comp_dir="${out_dir}_composited"

    if [ -d "$comp_dir" ] && [ "$(ls "$comp_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [GPU $gpu] [SKIP] ${LABEL}/${ds} inference (composited exist)"
        return 0
    fi

    local hp="exp_path=${EXP_PATH},ckpt_name=${LABEL}.pt,num_steps=${NUM_STEPS},guidance_scale=${GUIDANCE_SCALE},output_dir=${out_dir},latentsync_inference=true,use_mouth_only_compositing=true,face_detection_cache_dir=${face_cache},mask_all_frames=true,no_first_frame_overwrite=true,use_ref_sequence=true"

    echo "  [GPU $gpu] Inference ${LABEL}/${ds}..."
    mkdir -p "$out_dir"

    CUDA_VISIBLE_DEVICES=$gpu $TORCHRUN --standalone --nproc_per_node=1 \
        --master_port $((29500 + gpu)) \
        scripts/inference_v2v.py \
        --config configs/inference_v2v_1.3B.yaml \
        --input_file "$input_file" \
        -hp "$hp" \
        > "${out_dir}.log" 2>&1

    # Post-process: move composited outputs
    mkdir -p "$comp_dir"
    local moved=0
    for f in "$out_dir"/*_composited_audio.mp4; do
        [ -f "$f" ] || continue
        stem=$(basename "$f" _composited_audio.mp4)
        mv "$f" "$comp_dir/${stem}.mp4"
        moved=$((moved + 1))
    done

    # Move aligned outputs too (keep them, but not used for metrics)
    local aligned_dir="${out_dir}_aligned"
    mkdir -p "$aligned_dir"
    for f in "$out_dir"/*_aligned_audio.mp4; do
        [ -f "$f" ] || continue
        stem=$(basename "$f" _aligned_audio.mp4)
        mv "$f" "$aligned_dir/${stem}.mp4"
    done
    echo "  [GPU $gpu] [DONE] ${LABEL}/${ds}: ${moved} composited"
}

run_composited_metrics() {
    local gpu=$1 ds=$2
    local fake_dir="${PROJECT_DIR}/${OUTPUT_BASE}/${LABEL}/${ds}_composited"
    local real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals/${ds}"
    local log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics/${LABEL}/${ds}"

    if [ ! -d "$fake_dir" ] || [ "$(ls "$fake_dir"/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "  [GPU $gpu] [SKIP] composited metrics ${ds} (no outputs)"
        return
    fi

    if [ -f "$log_dir/metrics.log" ]; then
        echo "  [GPU $gpu] [SKIP] composited metrics ${ds} (results exist)"
        return
    fi

    rm -rf "$log_dir"
    mkdir -p "$log_dir"
    echo "  [GPU $gpu] Composited metrics ${LABEL}/${ds}..."

    cd "$METRICS_REPO"
    CUDA_VISIBLE_DEVICES=$gpu PATH="${CONDA_METRICS:+${CONDA_METRICS}:}$PATH" \
        bash eval/run_metrics.sh \
        --real_videos_dir "$real_dir" \
        --fake_videos_dir "$fake_dir" \
        --shape_predictor_path "$SHAPE_PRED" \
        --output_dir "$log_dir" --log_path "$log_dir/metrics.log" \
        --fallback_detection_confidence 0.2 --fake_videos_top_level --all \
        > "$log_dir/eval.log" 2>&1
    cd "$PROJECT_DIR"
    echo "  [GPU $gpu] [DONE] composited metrics ${LABEL}/${ds}"
}

run_aligned_crop_metrics() {
    local gpu=$1 ds=$2
    local fake_dir="${PROJECT_DIR}/${OUTPUT_BASE}/${LABEL}/${ds}_composited"
    local real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals/${ds}"
    local out_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics_ac/${LABEL}/${ds}"

    if [ ! -d "$fake_dir" ] || [ "$(ls "$fake_dir"/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "  [GPU $gpu] [SKIP] GT-aligned crop metrics ${ds} (no outputs)"
        return
    fi

    if [ -f "$out_dir/metrics_aligned.log" ]; then
        echo "  [GPU $gpu] [SKIP] GT-aligned crop metrics ${ds} (results exist)"
        return
    fi

    mkdir -p "$out_dir"
    echo "  [GPU $gpu] GT-aligned crop metrics ${LABEL}/${ds}..."

    CUDA_VISIBLE_DEVICES=$gpu $METRICS_PYTHON scripts/eval_aligned_crops.py \
        --real_videos_dir "$real_dir" \
        --fake_videos_dir "$fake_dir" \
        --output_dir "$out_dir" \
        > "$out_dir/eval.log" 2>&1
    echo "  [GPU $gpu] [DONE] GT-aligned crop metrics ${LABEL}/${ds}"
}

# ─── Per-GPU pipeline ────────────────────────────────────────────────────────
run_pipeline() {
    local gpu=$1 ds=$2 face_cache=$3
    run_inference "$gpu" "$ds" "$face_cache"
    run_composited_metrics "$gpu" "$ds"
    run_aligned_crop_metrics "$gpu" "$ds"
    echo "  [GPU $gpu] Pipeline complete: ${LABEL}/${ds}"
}

echo ""
echo "=== Running pipelined inference + metrics ==="
echo "  Metrics: composited + GT-aligned crops"

run_pipeline 2 hdtf "$FC_HDTF" &
pid2=$!
run_pipeline 3 hallo3 "$FC_HALLO3" &
pid3=$!

echo "  GPU 2 (PID $pid2): ${LABEL}/hdtf"
echo "  GPU 3 (PID $pid3): ${LABEL}/hallo3"

wait $pid2 || echo "WARNING: GPU 2 failed"
wait $pid3 || echo "WARNING: GPU 3 failed"

echo "  All pipelines complete."

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
DATASETS=(hdtf hallo3)

extract_metrics() {
    local log=$1
    [ -f "$log" ] || return 1
    fid=$(grep -oP 'mean_fid:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    ssim=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    fvd=$(grep -oP 'FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    csim=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    syncc=$(grep 'Mean SyncNet Confidence' "$log" 2>/dev/null | grep -oP '[\d.]+$' || echo "")
    syncd=$(grep 'Mean SyncNet Min Distance' "$log" 2>/dev/null | grep -oP '[\d.]+$' || echo "")
    lmd=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
}

extract_ac_metrics() {
    local log=$1
    [ -f "$log" ] || return 1
    ac_fid=$(grep -oP '^FID:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    ac_ssim=$(grep -oP '^SSIM:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    ac_fvd=$(grep -oP '^FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
}

echo "=== Metrics Summary (1.3B Phase 2) ==="
printf "%-14s %-10s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Label" "Dataset" "FID" "SSIM" "FVD" "CSIM" "Sync-C" "Sync-D" "LMD" "ac_FID" "ac_SSIM" "ac_FVD"
echo "--------------------------------------------------------------------------------------------------------------"
CSV_FILE="${OUTPUT_BASE}/metrics_summary.csv"
echo "method,cfg,dataset,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD,ac_FID,ac_SSIM,ac_FVD" > "$CSV_FILE"
for ds in "${DATASETS[@]}"; do
    log="${OUTPUT_BASE}/metrics/${LABEL}/${ds}/metrics.log"
    ac_log="${OUTPUT_BASE}/metrics_ac/${LABEL}/${ds}/metrics_aligned.log"
    fid="" ssim="" fvd="" csim="" syncc="" syncd="" lmd=""
    ac_fid="" ac_ssim="" ac_fvd=""
    extract_metrics "$log" 2>/dev/null || true
    extract_ac_metrics "$ac_log" 2>/dev/null || true
    printf "%-14s %-10s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n" \
        "$LABEL" "$ds" "${fid:--}" "${ssim:--}" "${fvd:--}" "${csim:--}" \
        "${syncc:--}" "${syncd:--}" "${lmd:--}" "${ac_fid:--}" "${ac_ssim:--}" "${ac_fvd:--}"
    echo "Phase2_1.3B_s${STEP},${GUIDANCE_SCALE},${ds},${fid},${ssim},${fvd},${csim},${syncc},${syncd},${lmd},${ac_fid},${ac_ssim},${ac_fvd}" >> "$CSV_FILE"
done
echo "CSV: ${CSV_FILE}"

echo ""
echo "Output: ${OUTPUT_BASE}/"
echo "Done."
