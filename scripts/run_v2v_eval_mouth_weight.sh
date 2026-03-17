#!/bin/bash
# Inference + metrics for 14B mouth_weight checkpoint.
# GPU 0: hdtf inference → hdtf metrics (composited + aligned)
# GPU 1: hallo3 inference → hallo3 metrics (composited + aligned)
# Designed to run alongside 1.3B training on GPUs 0,1 (eval is lightweight).
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
EXP_PATH="/home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_2gpu"
STEP=1500
NUM_STEPS=50
GUIDANCE_SCALE=4.5
OUTPUT_BASE="demo_out/v2v_eval_mouth_weight"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TORCHRUN="/home/work/.local/miniconda3/envs/omniavatar/bin/torchrun"
CONDA_METRICS="/home/work/.local/miniconda3/envs/latentsync-metrics/bin"
METRICS_REPO="/home/work/.local/latentsync-metrics"
SHAPE_PRED="${METRICS_REPO}/shape_predictor_68_face_landmarks.dat"
INPUT_DIR="/tmp/v2v_eval_inputs"

cd "$PROJECT_DIR"

# Face caches
FC_HDTF="/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data/face_cache"
FC_HALLO3="/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data/face_cache"

# GT originals (full resolution)
GT_HDTF="/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data"
GT_HALLO3="/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data"

# GT aligned (512x512)
GT_ALIGNED_HDTF="/home/work/.local/StableAvatar/latentsync_eval_hdtf/aligned_data"
GT_ALIGNED_HALLO3="/home/work/.local/StableAvatar/latentsync_eval_hallo3/aligned_data"

# ─── Verify ───────────────────────────────────────────────────────────────────
echo "=== Verification ==="
[ -f "${EXP_PATH}/step-${STEP}.pt" ] || { echo "ERROR: step-${STEP}.pt not found"; exit 1; }
echo "  step-${STEP}.pt OK"
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
        echo "  [SKIP] ${out_dir} exists"
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

# Full-res originals
setup_originals hdtf "$GT_HDTF" "${OUTPUT_BASE}/originals/hdtf"
setup_originals hallo3 "$GT_HALLO3" "${OUTPUT_BASE}/originals/hallo3"

# Aligned 512x512 originals
setup_originals hdtf_aligned "$GT_ALIGNED_HDTF" "${OUTPUT_BASE}/originals_aligned/hdtf"
setup_originals hallo3_aligned "$GT_ALIGNED_HALLO3" "${OUTPUT_BASE}/originals_aligned/hallo3"

# ─── Helper functions ─────────────────────────────────────────────────────────

run_inference() {
    local gpu=$1 ds=$2 face_cache=$3
    local out_dir="${OUTPUT_BASE}/step-${STEP}/${ds}"
    local input_file="${INPUT_DIR}/${ds}_latentsync.txt"
    local comp_dir="${out_dir}_composited"
    local aligned_dir="${out_dir}_aligned"

    if [ -d "$comp_dir" ] && [ "$(ls "$comp_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ] \
       && [ -d "$aligned_dir" ] && [ "$(ls "$aligned_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [GPU $gpu] [SKIP] step-${STEP}/${ds} (composited + aligned exist)"
        return 0
    fi

    local hp="exp_path=${EXP_PATH},ckpt_name=step-${STEP}.pt,num_steps=${NUM_STEPS},guidance_scale=${GUIDANCE_SCALE},output_dir=${out_dir},latentsync_inference=true,use_mouth_only_compositing=true,face_detection_cache_dir=${face_cache},mask_all_frames=true,no_first_frame_overwrite=true,use_ref_sequence=true"

    echo "  [GPU $gpu] Inference step-${STEP}/${ds}..."
    mkdir -p "$out_dir"

    CUDA_VISIBLE_DEVICES=$gpu $TORCHRUN --standalone --nproc_per_node=1 \
        --master_port $((29500 + gpu)) \
        scripts/inference_v2v.py \
        --config configs/inference_v2v.yaml \
        --input_file "$input_file" \
        -hp "$hp" \
        > "${out_dir}.log" 2>&1

    # Post-process: composited outputs
    mkdir -p "$comp_dir"
    local moved=0
    for f in "$out_dir"/*_composited_audio.mp4; do
        [ -f "$f" ] || continue
        stem=$(basename "$f" _composited_audio.mp4)
        mv "$f" "$comp_dir/${stem}.mp4"
        moved=$((moved + 1))
    done

    # Post-process: aligned outputs
    mkdir -p "$aligned_dir"
    local aligned_moved=0
    for f in "$out_dir"/*_aligned_audio.mp4; do
        [ -f "$f" ] || continue
        stem=$(basename "$f" _aligned_audio.mp4)
        mv "$f" "$aligned_dir/${stem}.mp4"
        aligned_moved=$((aligned_moved + 1))
    done
    echo "  [GPU $gpu] [DONE] step-${STEP}/${ds}: ${moved} composited, ${aligned_moved} aligned"
}

run_metrics_standard() {
    local gpu=$1 ds=$2
    local fake_dir="${PROJECT_DIR}/${OUTPUT_BASE}/step-${STEP}/${ds}_composited"
    local real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals/${ds}"
    local log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics/step-${STEP}/${ds}"

    if [ ! -d "$fake_dir" ] || [ "$(ls "$fake_dir"/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "  [GPU $gpu] [SKIP] composited metrics ${ds} (no outputs)"
        return
    fi

    rm -rf "$log_dir"
    mkdir -p "$log_dir"
    echo "  [GPU $gpu] Metrics (composited) step-${STEP}/${ds}..."

    cd "$METRICS_REPO"
    CUDA_VISIBLE_DEVICES=$gpu PATH="${CONDA_METRICS}:$PATH" \
        bash eval/run_metrics.sh \
        --real_videos_dir "$real_dir" \
        --fake_videos_dir "$fake_dir" \
        --shape_predictor_path "$SHAPE_PRED" \
        --output_dir "$log_dir" --log_path "$log_dir/metrics.log" \
        --fallback_detection_confidence 0.2 --fake_videos_top_level --all \
        > "$log_dir/eval.log" 2>&1
    cd "$PROJECT_DIR"
    echo "  [GPU $gpu] [DONE] composited metrics ${ds}"
}

run_metrics_aligned() {
    local gpu=$1 ds=$2
    local fake_dir="${PROJECT_DIR}/${OUTPUT_BASE}/step-${STEP}/${ds}_aligned"
    local real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals_aligned/${ds}"
    local log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics_aligned/step-${STEP}/${ds}"

    if [ ! -d "$fake_dir" ] || [ "$(ls "$fake_dir"/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "  [GPU $gpu] [SKIP] aligned metrics ${ds} (no outputs)"
        return
    fi

    rm -rf "$log_dir"
    mkdir -p "$log_dir"
    echo "  [GPU $gpu] Metrics (aligned) step-${STEP}/${ds}..."

    cd "$METRICS_REPO"
    CUDA_VISIBLE_DEVICES=$gpu PATH="${CONDA_METRICS}:$PATH" \
        bash eval/run_metrics.sh \
        --real_videos_dir "$real_dir" \
        --fake_videos_dir "$fake_dir" \
        --shape_predictor_path "$SHAPE_PRED" \
        --output_dir "$log_dir" --log_path "$log_dir/metrics.log" \
        --fallback_detection_confidence 0.2 --fake_videos_top_level --all \
        > "$log_dir/eval.log" 2>&1
    cd "$PROJECT_DIR"
    echo "  [GPU $gpu] [DONE] aligned metrics ${ds}"
}

# ─── Phase 1: Inference (parallel on 2 GPUs) ─────────────────────────────────
echo ""
echo "=== Phase 1: Inference (step-${STEP}) ==="
run_inference 0 hdtf "$FC_HDTF" &
run_inference 1 hallo3 "$FC_HALLO3" &
wait
echo "  Phase 1 complete."

# ─── Launch 1.3B training in background ───────────────────────────────────────
echo ""
echo "=== Launching 1.3B training on GPUs 0,1 ==="
TRAIN_LOG="${PROJECT_DIR}/demo_out/v2v_eval_mouth_weight/1.3B_training.log"
mkdir -p "$(dirname "$TRAIN_LOG")"
nohup bash "${PROJECT_DIR}/scripts/continue_v2v_1.3B_maskall_refseq_2gpus_mouth_weight.sh" \
    > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "  1.3B training PID: ${TRAIN_PID} (log: ${TRAIN_LOG})"
echo "  Waiting 30s for training to initialize before running metrics..."
sleep 30

# ─── Phase 2: Metrics — composited (parallel on 2 GPUs) ──────────────────────
# Metrics use ~2-4 GB GPU (FID/FVD/SyncNet models). 1.3B training uses ~40-50 GB.
# On 144 GB H200s, this should fit comfortably.
echo ""
echo "=== Phase 2: Composited Metrics ==="
run_metrics_standard 0 hdtf &
run_metrics_standard 1 hallo3 &
wait
echo "  Phase 2 complete."

# ─── Phase 3: Metrics — aligned (parallel on 2 GPUs) ─────────────────────────
echo ""
echo "=== Phase 3: Aligned Metrics ==="
run_metrics_aligned 0 hdtf &
run_metrics_aligned 1 hallo3 &
wait
echo "  Phase 3 complete."

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Results: step-${STEP} ==="
echo ""
echo "--- Composited (original resolution) ---"
printf "%-10s %8s %8s %8s %8s %8s %8s %8s\n" "Dataset" "FID" "SSIM" "FVD" "CSIM" "Sync-C" "Sync-D" "LMD"
for ds in hdtf hallo3; do
    dir="${OUTPUT_BASE}/metrics/step-${STEP}/${ds}"
    fid=$(grep "mean_fid:" "$dir/fid_per_video.log" 2>/dev/null | awk '{print $2}')
    ssim=$(grep "mean_ssim:" "$dir/ssim_lmd_per_video.log" 2>/dev/null | awk '{print $2}')
    lmd=$(grep "mean_lmd:" "$dir/ssim_lmd_per_video.log" 2>/dev/null | awk '{print $2}')
    fvd=$(grep "^FVD:" "$dir/metrics.log" 2>/dev/null | awk '{print $2}')
    csim=$(grep "^CSIM:" "$dir/metrics.log" 2>/dev/null | awk '{print $2}')
    syncc=$(grep "Mean SyncNet Confidence" "$dir/metrics.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    syncd=$(grep "Mean SyncNet Min Distance" "$dir/metrics.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-10s %8s %8s %8s %8s %8s %8s %8s\n" \
        "$ds" "${fid:--}" "${ssim:--}" "${fvd:--}" "${csim:--}" "${syncc:--}" "${syncd:--}" "${lmd:--}"
done

echo ""
echo "--- Aligned (512x512 GT-aligned crops) ---"
printf "%-10s %8s %8s %8s %8s %8s %8s %8s\n" "Dataset" "FID" "SSIM" "FVD" "CSIM" "Sync-C" "Sync-D" "LMD"
for ds in hdtf hallo3; do
    dir="${OUTPUT_BASE}/metrics_aligned/step-${STEP}/${ds}"
    fid=$(grep "mean_fid:" "$dir/fid_per_video.log" 2>/dev/null | awk '{print $2}')
    ssim=$(grep "mean_ssim:" "$dir/ssim_lmd_per_video.log" 2>/dev/null | awk '{print $2}')
    lmd=$(grep "mean_lmd:" "$dir/ssim_lmd_per_video.log" 2>/dev/null | awk '{print $2}')
    fvd=$(grep "^FVD:" "$dir/metrics.log" 2>/dev/null | awk '{print $2}')
    csim=$(grep "^CSIM:" "$dir/metrics.log" 2>/dev/null | awk '{print $2}')
    syncc=$(grep "Mean SyncNet Confidence" "$dir/metrics.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    syncd=$(grep "Mean SyncNet Min Distance" "$dir/metrics.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    printf "%-10s %8s %8s %8s %8s %8s %8s %8s\n" \
        "$ds" "${fid:--}" "${ssim:--}" "${fvd:--}" "${csim:--}" "${syncc:--}" "${syncd:--}" "${lmd:--}"
done

echo ""
echo "=== Done ==="
