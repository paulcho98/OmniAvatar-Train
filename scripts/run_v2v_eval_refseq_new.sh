#!/bin/bash
# Pipelined inference + metrics (composited + aligned) + stitching for refseq checkpoints.
# Each GPU runs its dataset end-to-end: inference → composited metrics → aligned metrics → stitch.
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
EXP_PATH="/home/work/output_omniavatar_v2v_maskall_refseq_new_data_loss_weights"
STEPS=(3000)
NUM_STEPS=50
GUIDANCE_SCALE=4.5
OUTPUT_BASE="demo_out/v2v_eval_refseq_new"
INPUT_DIR="/tmp/v2v_eval_inputs"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TORCHRUN="/home/work/.local/miniconda3/envs/omniavatar/bin/torchrun"
CONDA_METRICS="/home/work/.local/miniconda3/envs/latentsync-metrics/bin"
METRICS_REPO="/home/work/.local/latentsync-metrics"
SHAPE_PRED="${METRICS_REPO}/shape_predictor_68_face_landmarks.dat"

cd "$PROJECT_DIR"

# Face caches
FC_HDTF="/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data/face_cache"
FC_HALLO3="/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data/face_cache"
FC_MIXED="/home/work/.local/StableAvatar/validation_hallo3_mixed/shared_data/face_cache"

# GT originals (full resolution)
GT_HDTF="/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data"
GT_HALLO3="/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data"
GT_MIXED="/home/work/.local/StableAvatar/validation_hallo3_mixed/shared_data"

# GT aligned (512x512)
GT_ALIGNED_HDTF="/home/work/.local/StableAvatar/latentsync_eval_hdtf/aligned_data"
GT_ALIGNED_HALLO3="/home/work/.local/StableAvatar/latentsync_eval_hallo3/aligned_data"
GT_ALIGNED_MIXED="/home/work/.local/StableAvatar/latentsync_eval_hallo3_mixed/aligned_data"

# Maskall composited (for 3-way stitching)
MASKALL_BASE="demo_out/v2v_eval_maskall/step-5500"

# ─── Verify ───────────────────────────────────────────────────────────────────
echo "=== Verification ==="
for step in "${STEPS[@]}"; do
    [ -f "${EXP_PATH}/step-${step}.pt" ] || { echo "ERROR: step-${step}.pt not found"; exit 1; }
    echo "  step-${step}.pt OK"
done
for ds in hdtf hallo3 hallo3_mixed; do
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
setup_originals hallo3_mixed "$GT_MIXED" "${OUTPUT_BASE}/originals/hallo3_mixed"

# Aligned 512x512 originals
setup_originals hdtf_aligned "$GT_ALIGNED_HDTF" "${OUTPUT_BASE}/originals_aligned/hdtf"
setup_originals hallo3_aligned "$GT_ALIGNED_HALLO3" "${OUTPUT_BASE}/originals_aligned/hallo3"
setup_originals hallo3_mixed_aligned "$GT_ALIGNED_MIXED" "${OUTPUT_BASE}/originals_aligned/hallo3_mixed"

# ─── Helper functions ─────────────────────────────────────────────────────────

run_inference() {
    local gpu=$1 step=$2 ds=$3 face_cache=$4
    local out_dir="${OUTPUT_BASE}/step-${step}/${ds}"
    local input_file="${INPUT_DIR}/${ds}_latentsync.txt"
    local comp_dir="${out_dir}_composited"
    local aligned_dir="${out_dir}_aligned"

    if [ -d "$comp_dir" ] && [ "$(ls "$comp_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ] \
       && [ -d "$aligned_dir" ] && [ "$(ls "$aligned_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [GPU $gpu] [SKIP] step-${step}/${ds} (composited + aligned exist)"
        return 0
    fi

    local hp="exp_path=${EXP_PATH},ckpt_name=step-${step}.pt,num_steps=${NUM_STEPS},guidance_scale=${GUIDANCE_SCALE},output_dir=${out_dir},latentsync_inference=true,use_mouth_only_compositing=true,face_detection_cache_dir=${face_cache},mask_all_frames=true,no_first_frame_overwrite=true,use_ref_sequence=true"

    echo "  [GPU $gpu] Inference step-${step}/${ds}..."
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
    echo "  [GPU $gpu] [DONE] step-${step}/${ds}: ${moved} composited, ${aligned_moved} aligned"
}

run_metrics() {
    local gpu=$1 step=$2 ds=$3 type=$4
    local fake_dir real_dir log_dir

    if [ "$type" = "composited" ]; then
        fake_dir="${PROJECT_DIR}/${OUTPUT_BASE}/step-${step}/${ds}_composited"
        real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals/${ds}"
        log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics/step-${step}/${ds}"
    else
        fake_dir="${PROJECT_DIR}/${OUTPUT_BASE}/step-${step}/${ds}_aligned"
        real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals_aligned/${ds}"
        log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics_aligned/step-${step}/${ds}"
    fi

    if [ ! -d "$fake_dir" ] || [ "$(ls "$fake_dir"/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "  [GPU $gpu] [SKIP] ${type} metrics step-${step}/${ds} (no outputs)"
        return
    fi

    rm -rf "$log_dir"
    mkdir -p "$log_dir"
    echo "  [GPU $gpu] Metrics (${type}) step-${step}/${ds}..."

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
    echo "  [GPU $gpu] [DONE] ${type} metrics step-${step}/${ds}"
}

run_stitch() {
    local step=$1 ds=$2
    # Skip non-recon datasets
    [ "$ds" = "hdtf" ] || [ "$ds" = "hallo3" ] || return 0

    local gt_base comp_dir stitch_dir
    case "$ds" in
        hdtf) gt_base="$GT_HDTF" ;;
        hallo3) gt_base="$GT_HALLO3" ;;
    esac
    comp_dir="${OUTPUT_BASE}/step-${step}/${ds}_composited"
    [ -d "$comp_dir" ] || return 0

    # 2-way stitch: GT | refseq
    stitch_dir="${OUTPUT_BASE}/stitched/step-${step}/${ds}"
    echo "  Stitching 2-way step-${step}/${ds}..."
    for gen_video in "$comp_dir"/*.mp4; do
        local name=$(basename "$gen_video" .mp4)
        local gt_video="${gt_base}/${name}/sub_clip.mp4"
        local out_video="${stitch_dir}/${name}.mp4"
        [ -f "$gt_video" ] || continue
        [ -f "$out_video" ] && continue
        local gt_h=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$gt_video")
        gt_h=$(( (gt_h / 2) * 2 ))
        mkdir -p "$stitch_dir"
        ffmpeg -i "$gt_video" -i "$gen_video" \
            -filter_complex "[0:v]scale=-2:${gt_h}[gt];[1:v]scale=-2:${gt_h}[gen];[gt][gen]hstack=inputs=2" \
            -c:v libx264 -preset fast -crf 18 -c:a aac -map 1:a? -shortest -y \
            "$out_video" </dev/null 2>/dev/null
    done

    # 3-way stitch: GT | maskall-5500 | refseq
    local maskall_dir="${MASKALL_BASE}/${ds}_composited"
    [ -d "$maskall_dir" ] || return 0
    local stitch3_dir="${OUTPUT_BASE}/stitched_mask_ref_compare/${ds}"
    echo "  Stitching 3-way step-${step}/${ds}..."
    for gen_video in "$comp_dir"/*.mp4; do
        local name=$(basename "$gen_video" .mp4)
        local gt_video="${gt_base}/${name}/sub_clip.mp4"
        local maskall_video="${maskall_dir}/${name}.mp4"
        local out_video="${stitch3_dir}/${name}.mp4"
        [ -f "$gt_video" ] && [ -f "$maskall_video" ] || continue
        [ -f "$out_video" ] && continue
        local gt_h=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$gt_video")
        gt_h=$(( (gt_h / 2) * 2 ))
        mkdir -p "$stitch3_dir"
        ffmpeg -i "$gt_video" -i "$maskall_video" -i "$gen_video" \
            -filter_complex "[0:v]scale=-2:${gt_h}[a];[1:v]scale=-2:${gt_h}[b];[2:v]scale=-2:${gt_h}[c];[a][b][c]hstack=inputs=3" \
            -c:v libx264 -preset fast -crf 18 -c:a aac -map 2:a? -shortest -y \
            "$out_video" </dev/null 2>/dev/null
    done
}

# ─── Per-GPU pipeline ─────────────────────────────────────────────────────────
run_pipeline() {
    local gpu=$1 step=$2 ds=$3 face_cache=$4
    run_inference "$gpu" "$step" "$ds" "$face_cache"
    run_metrics "$gpu" "$step" "$ds" composited
    run_metrics "$gpu" "$step" "$ds" aligned
    run_stitch "$step" "$ds"
    echo "  [GPU $gpu] Pipeline complete: step-${step}/${ds}"
}

echo ""
echo "=== Running pipelined inference + metrics + stitching ==="

for step in "${STEPS[@]}"; do
    run_pipeline 0 "$step" hdtf "$FC_HDTF" &
    pid0=$!
    run_pipeline 1 "$step" hallo3 "$FC_HALLO3" &
    pid1=$!
    run_pipeline 2 "$step" hallo3_mixed "$FC_MIXED" &
    pid2=$!

    echo "  GPU 0 (PID $pid0): step-${step}/hdtf"
    echo "  GPU 1 (PID $pid1): step-${step}/hallo3"
    echo "  GPU 2 (PID $pid2): step-${step}/hallo3_mixed"

    wait $pid0 || echo "WARNING: GPU 0 failed"
    wait $pid1 || echo "WARNING: GPU 1 failed"
    wait $pid2 || echo "WARNING: GPU 2 failed"
done

echo "  All pipelines complete."

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
DATASETS=(hdtf hallo3 hallo3_mixed)

ALL_STEPS=()
for d in "${OUTPUT_BASE}"/metrics/step-*/; do
    [ -d "$d" ] || continue
    ALL_STEPS+=("$(basename "$d" | sed 's/step-//')")
done
IFS=$'\n' ALL_STEPS=($(sort -n <<<"${ALL_STEPS[*]}")); unset IFS

extract_metrics() {
    local log=$1
    [ -f "$log" ] || return 1
    fid=$(grep -oP 'mean_fid:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    ssim=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    fvd=$(grep -oP 'FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    csim=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    syncc=$(grep -oP 'Sync-C.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    syncd=$(grep -oP 'Sync-D.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    lmd=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
}

# Composited metrics
echo "=== Composited Metrics Summary ==="
printf "%-10s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Step" "Dataset" "FID" "SSIM" "FVD" "CSIM" "Sync-C" "Sync-D" "LMD"
echo "------------------------------------------------------------------------------------"
CSV_FILE="${OUTPUT_BASE}/metrics_summary.csv"
echo "step,dataset,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD" > "$CSV_FILE"
for step in "${ALL_STEPS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        log="${OUTPUT_BASE}/metrics/step-${step}/${ds}/metrics.log"
        extract_metrics "$log" || continue
        printf "%-10s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
            "step-${step}" "$ds" "${fid:--}" "${ssim:--}" "${fvd:--}" "${csim:--}" "${syncc:--}" "${syncd:--}" "${lmd:--}"
        echo "step-${step},${ds},${fid},${ssim},${fvd},${csim},${syncc},${syncd},${lmd}" >> "$CSV_FILE"
    done
done
echo "CSV: ${CSV_FILE}"

# Aligned metrics
echo ""
echo "=== Aligned Metrics Summary (512x512) ==="
printf "%-10s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Step" "Dataset" "FID" "SSIM" "FVD" "CSIM" "Sync-C" "Sync-D" "LMD"
echo "------------------------------------------------------------------------------------"
ALIGNED_CSV="${OUTPUT_BASE}/metrics_aligned_summary.csv"
echo "step,dataset,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD" > "$ALIGNED_CSV"
for step in "${ALL_STEPS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        log="${OUTPUT_BASE}/metrics_aligned/step-${step}/${ds}/metrics.log"
        extract_metrics "$log" || continue
        printf "%-10s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
            "step-${step}" "$ds" "${fid:--}" "${ssim:--}" "${fvd:--}" "${csim:--}" "${syncc:--}" "${syncd:--}" "${lmd:--}"
        echo "step-${step},${ds},${fid},${ssim},${fvd},${csim},${syncc},${syncd},${lmd}" >> "$ALIGNED_CSV"
    done
done
echo "CSV: ${ALIGNED_CSV}"

echo ""
echo "Output: ${OUTPUT_BASE}/"
echo "Done."
