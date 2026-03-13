#!/bin/bash
# Inference + metrics + stitching for mask_all_frames checkpoint.
# Uses step-3000 from /home/work/output_omniavatar_v2v_maskall/
# Flags: mask_all_frames=true, no_first_frame_overwrite=true
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
EXP_PATH="/home/work/output_omniavatar_v2v_maskall"
CKPT_NAME="step-3000.pt"
NUM_STEPS=50
GUIDANCE_SCALE=4.5
OUTPUT_BASE="demo_out/v2v_eval_maskall"
INPUT_DIR="/tmp/v2v_eval_inputs"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TORCHRUN="/home/work/.local/miniconda3/envs/omniavatar/bin/torchrun"
PYTHON="/home/work/.local/miniconda3/envs/omniavatar/bin/python"
CONDA_METRICS="/home/work/.local/miniconda3/envs/latentsync-metrics/bin"
METRICS_REPO="/home/work/.local/latentsync-metrics"
SHAPE_PRED="${METRICS_REPO}/shape_predictor_68_face_landmarks.dat"

cd "$PROJECT_DIR"

# Face caches
FC_HDTF="/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data/face_cache"
FC_HALLO3="/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data/face_cache"
FC_MIXED="/home/work/.local/StableAvatar/validation_hallo3_mixed/shared_data/face_cache"

# GT originals (for metrics)
GT_HDTF="/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data"
GT_HALLO3="/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data"
GT_MIXED="/home/work/.local/StableAvatar/validation_hallo3_mixed/shared_data"

# ─── Verify inputs exist ─────────────────────────────────────────────────────
echo "=== Verification ==="
[ -f "${EXP_PATH}/${CKPT_NAME}" ] || { echo "ERROR: ${EXP_PATH}/${CKPT_NAME} not found"; exit 1; }
echo "  Checkpoint: ${EXP_PATH}/${CKPT_NAME}"
for ds in hdtf hallo3 hallo3_mixed; do
    [ -f "${INPUT_DIR}/${ds}_latentsync.txt" ] || { echo "ERROR: ${INPUT_DIR}/${ds}_latentsync.txt not found"; exit 1; }
done
echo "  Input files OK"

# ─── Setup originals symlinks ────────────────────────────────────────────────
echo ""
echo "=== Setting up originals ==="
for ds_info in "hdtf|${GT_HDTF}" "hallo3|${GT_HALLO3}" "hallo3_mixed|${GT_MIXED}"; do
    IFS='|' read -r ds gt_dir <<< "$ds_info"
    orig_dir="${OUTPUT_BASE}/originals/${ds}"
    mkdir -p "$orig_dir"
    if [ "$(ls "$orig_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] ${ds} originals already set up"
        continue
    fi
    for d in "$gt_dir"/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        [ "$name" = "face_cache" ] && continue
        [ -f "$d/sub_clip.mp4" ] && ln -sf "$d/sub_clip.mp4" "$orig_dir/${name}.mp4"
    done
    count=$(ls "$orig_dir"/*.mp4 2>/dev/null | wc -l)
    echo "  ${ds}: ${count} originals linked"
done

# ─── Inference ────────────────────────────────────────────────────────────────
echo ""
echo "=== Inference (3 datasets, GPUs 0-2) ==="

run_inference() {
    local gpu=$1 ds=$2 face_cache=$3
    local out_dir="${OUTPUT_BASE}/${ds}"
    local input_file="${INPUT_DIR}/${ds}_latentsync.txt"
    local comp_dir="${out_dir}_composited"

    # Skip if already done
    if [ -d "$comp_dir" ] && [ "$(ls "$comp_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] ${ds} (composited exists)"
        return 0
    fi

    local hp="exp_path=${EXP_PATH},ckpt_name=${CKPT_NAME},num_steps=${NUM_STEPS},guidance_scale=${GUIDANCE_SCALE},output_dir=${out_dir},latentsync_inference=true,use_mouth_only_compositing=true,face_detection_cache_dir=${face_cache},mask_all_frames=true,no_first_frame_overwrite=true"

    echo "  [GPU $gpu] ${ds}..."
    mkdir -p "$out_dir"

    CUDA_VISIBLE_DEVICES=$gpu $TORCHRUN --standalone --nproc_per_node=1 \
        --master_port $((29500 + gpu)) \
        scripts/inference_v2v.py \
        --config configs/inference_v2v.yaml \
        --input_file "$input_file" \
        -hp "$hp" \
        > "${out_dir}.log" 2>&1

    # Post-process: move composited videos
    mkdir -p "$comp_dir"
    local moved=0
    for f in "$out_dir"/*_composited_audio.mp4; do
        [ -f "$f" ] || continue
        stem=$(basename "$f" _composited_audio.mp4)
        mv "$f" "$comp_dir/${stem}.mp4"
        moved=$((moved + 1))
    done
    echo "  [DONE] ${ds}: ${moved} videos composited"
}

run_inference 0 hdtf "$FC_HDTF" &
pid0=$!
run_inference 1 hallo3 "$FC_HALLO3" &
pid1=$!
run_inference 2 hallo3_mixed "$FC_MIXED" &
pid2=$!

echo "  GPU 0 (PID $pid0): hdtf (33 samples)"
echo "  GPU 1 (PID $pid1): hallo3 (30 samples)"
echo "  GPU 2 (PID $pid2): hallo3_mixed (12 samples)"

wait $pid0 || echo "WARNING: GPU 0 had failures"
wait $pid1 || echo "WARNING: GPU 1 had failures"
wait $pid2 || echo "WARNING: GPU 2 had failures"
echo "  All inference complete."

# ─── Metrics ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Metrics ==="
cd "$METRICS_REPO"

DATASETS=(hdtf hallo3 hallo3_mixed)
GPUS=(0 1 2)
pids=()

for i in "${!DATASETS[@]}"; do
    ds="${DATASETS[$i]}"
    gpu="${GPUS[$i]}"
    comp_dir="${PROJECT_DIR}/${OUTPUT_BASE}/${ds}_composited"
    real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals/${ds}"
    log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics/${ds}"

    if [ ! -d "$comp_dir" ] || [ "$(ls "$comp_dir"/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "  [SKIP] metrics ${ds} (no composited videos)"
        continue
    fi

    rm -rf "$log_dir"
    mkdir -p "$log_dir"
    echo "  [GPU $gpu] metrics ${ds}"

    CUDA_VISIBLE_DEVICES=$gpu PATH="${CONDA_METRICS}:$PATH" \
        bash eval/run_metrics.sh \
        --real_videos_dir "$real_dir" \
        --fake_videos_dir "$comp_dir" \
        --shape_predictor_path "$SHAPE_PRED" \
        --output_dir "$log_dir" --log_path "$log_dir/metrics.log" \
        --fallback_detection_confidence 0.2 --fake_videos_top_level --all \
        > "$log_dir/eval.log" 2>&1 &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid" || echo "WARNING: metrics PID $pid failed"
done

cd "$PROJECT_DIR"
echo "  All metrics complete."

# ─── Stitching (recon only) ──────────────────────────────────────────────────
echo ""
echo "=== Stitching GT vs Generated ==="

stitch_one() {
    local gt_video=$1 gen_video=$2 out_video=$3
    local gt_h
    gt_h=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$gt_video")
    gt_h=$(( (gt_h / 2) * 2 ))
    mkdir -p "$(dirname "$out_video")"
    ffmpeg -i "$gt_video" -i "$gen_video" \
        -filter_complex "[0:v]scale=-2:${gt_h}[gt];[1:v]scale=-2:${gt_h}[gen];[gt][gen]hstack=inputs=2" \
        -c:v libx264 -preset fast -crf 18 -c:a aac -map 1:a? -shortest -y \
        "$out_video" </dev/null 2>/dev/null
}

for ds in hdtf hallo3; do
    comp_dir="${OUTPUT_BASE}/${ds}_composited"
    stitch_dir="${OUTPUT_BASE}/stitched/${ds}"
    case "$ds" in
        hdtf) gt_base="$GT_HDTF" ;;
        hallo3) gt_base="$GT_HALLO3" ;;
    esac

    [ -d "$comp_dir" ] || continue
    echo "  Stitching ${ds}..."
    running=0
    for gen_video in "$comp_dir"/*.mp4; do
        name=$(basename "$gen_video" .mp4)
        gt_video="${gt_base}/${name}/sub_clip.mp4"
        out_video="${stitch_dir}/${name}.mp4"
        [ -f "$gt_video" ] || continue
        [ -f "$out_video" ] && continue
        stitch_one "$gt_video" "$gen_video" "$out_video" &
        running=$((running + 1))
        if [ "$running" -ge 4 ]; then
            wait -n 2>/dev/null || true
            running=$((running - 1))
        fi
    done
    wait
    count=$(ls "$stitch_dir"/*.mp4 2>/dev/null | wc -l)
    echo "  ${ds}: ${count} stitched"
done

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Metrics Summary (maskall step-3000) ==="
printf "%-14s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Dataset" "FVD" "FID" "CSIM" "SSIM" "LMD" "Sync-D" "Sync-C"
echo "------------------------------------------------------------------------"
for ds in "${DATASETS[@]}"; do
    log="${OUTPUT_BASE}/metrics/${ds}/metrics.log"
    [ -f "$log" ] || continue
    fvd=$(grep -oP 'FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
    fid=$(grep -oP 'mean_fid:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
    csim=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
    ssim=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
    lmd=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
    syncd=$(grep -oP 'Sync-D.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
    syncc=$(grep -oP 'Sync-C.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
    printf "%-14s %8s %8s %8s %8s %8s %8s %8s\n" \
        "$ds" "$fvd" "$fid" "$csim" "$ssim" "$lmd" "$syncd" "$syncc"
done
echo ""
echo "Output: ${OUTPUT_BASE}/"
echo "Done."
