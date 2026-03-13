#!/bin/bash
# Re-run hallo3_mixed with updated float compositing for A/B comparison.
# Single GPU (2), single dataset, same checkpoint as refseq_new.
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
GPU=2
EXP_PATH="/home/work/output_omniavatar_v2v_maskall_refseq_new_data_loss_weights"
STEP=3000
NUM_STEPS=50
GUIDANCE_SCALE=4.5
OUTPUT_BASE="demo_out/v2v_eval_refseq_new_float"
INPUT_DIR="/tmp/v2v_eval_inputs"
DS="hallo3_mixed"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TORCHRUN="/home/work/.local/miniconda3/envs/omniavatar/bin/torchrun"
CONDA_METRICS="/home/work/.local/miniconda3/envs/latentsync-metrics/bin"
METRICS_REPO="/home/work/.local/latentsync-metrics"
SHAPE_PRED="${METRICS_REPO}/shape_predictor_68_face_landmarks.dat"

cd "$PROJECT_DIR"

FC="/home/work/.local/StableAvatar/validation_hallo3_mixed/shared_data/face_cache"
GT="/home/work/.local/StableAvatar/validation_hallo3_mixed/shared_data"
GT_ALIGNED="/home/work/.local/StableAvatar/latentsync_eval_hallo3_mixed/aligned_data"

# ─── Verify ───────────────────────────────────────────────────────────────────
echo "=== Verification ==="
[ -f "${EXP_PATH}/step-${STEP}.pt" ] || { echo "ERROR: step-${STEP}.pt not found"; exit 1; }
[ -f "${INPUT_DIR}/${DS}_latentsync.txt" ] || { echo "ERROR: input file missing"; exit 1; }
echo "  OK"

# ─── Setup originals ─────────────────────────────────────────────────────────
for info in "originals|${GT}" "originals_aligned|${GT_ALIGNED}"; do
    IFS='|' read -r dir_name gt_dir <<< "$info"
    orig_dir="${OUTPUT_BASE}/${dir_name}/${DS}"
    mkdir -p "$orig_dir"
    if [ "$(ls "$orig_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] ${dir_name}/${DS} exists"
        continue
    fi
    for d in "$gt_dir"/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        [ "$name" = "face_cache" ] && continue
        [ -f "$d/sub_clip.mp4" ] && ln -sf "$d/sub_clip.mp4" "$orig_dir/${name}.mp4"
    done
    echo "  ${dir_name}/${DS}: $(ls "$orig_dir"/*.mp4 | wc -l) linked"
done

# ─── Inference ────────────────────────────────────────────────────────────────
echo ""
echo "=== Inference (GPU ${GPU}, ${DS}) ==="
OUT_DIR="${OUTPUT_BASE}/step-${STEP}/${DS}"
COMP_DIR="${OUT_DIR}_composited"
ALIGNED_DIR="${OUT_DIR}_aligned"

if [ -d "$COMP_DIR" ] && [ "$(ls "$COMP_DIR"/*.mp4 2>/dev/null | wc -l)" -gt 0 ] \
   && [ -d "$ALIGNED_DIR" ] && [ "$(ls "$ALIGNED_DIR"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  [SKIP] already done"
else
    HP="exp_path=${EXP_PATH},ckpt_name=step-${STEP}.pt,num_steps=${NUM_STEPS},guidance_scale=${GUIDANCE_SCALE},output_dir=${OUT_DIR},latentsync_inference=true,use_mouth_only_compositing=true,face_detection_cache_dir=${FC},mask_all_frames=true,no_first_frame_overwrite=true,use_ref_sequence=true"

    mkdir -p "$OUT_DIR"
    CUDA_VISIBLE_DEVICES=$GPU $TORCHRUN --standalone --nproc_per_node=1 \
        --master_port $((29500 + GPU)) \
        scripts/inference_v2v.py \
        --config configs/inference_v2v.yaml \
        --input_file "${INPUT_DIR}/${DS}_latentsync.txt" \
        -hp "$HP" \
        > "${OUT_DIR}.log" 2>&1

    # Post-process composited
    mkdir -p "$COMP_DIR"
    for f in "$OUT_DIR"/*_composited_audio.mp4; do
        [ -f "$f" ] || continue
        mv "$f" "$COMP_DIR/$(basename "$f" _composited_audio.mp4).mp4"
    done

    # Post-process aligned
    mkdir -p "$ALIGNED_DIR"
    for f in "$OUT_DIR"/*_aligned_audio.mp4; do
        [ -f "$f" ] || continue
        mv "$f" "$ALIGNED_DIR/$(basename "$f" _aligned_audio.mp4).mp4"
    done
    echo "  Done: $(ls "$COMP_DIR"/*.mp4 2>/dev/null | wc -l) composited, $(ls "$ALIGNED_DIR"/*.mp4 2>/dev/null | wc -l) aligned"
fi

# ─── Metrics ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Metrics ==="
cd "$METRICS_REPO"

for type in composited aligned; do
    if [ "$type" = "composited" ]; then
        fake_dir="${PROJECT_DIR}/${COMP_DIR}"
        real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals/${DS}"
        log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics/step-${STEP}/${DS}"
    else
        fake_dir="${PROJECT_DIR}/${ALIGNED_DIR}"
        real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals_aligned/${DS}"
        log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics_aligned/step-${STEP}/${DS}"
    fi

    if [ ! -d "$fake_dir" ] || [ "$(ls "$fake_dir"/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "  [SKIP] ${type} metrics (no outputs)"
        continue
    fi

    rm -rf "$log_dir"
    mkdir -p "$log_dir"
    echo "  [GPU $GPU] ${type} metrics..."

    CUDA_VISIBLE_DEVICES=$GPU PATH="${CONDA_METRICS}:$PATH" \
        bash eval/run_metrics.sh \
        --real_videos_dir "$real_dir" \
        --fake_videos_dir "$fake_dir" \
        --shape_predictor_path "$SHAPE_PRED" \
        --output_dir "$log_dir" --log_path "$log_dir/metrics.log" \
        --fallback_detection_confidence 0.2 --fake_videos_top_level --all \
        > "$log_dir/eval.log" 2>&1
    echo "  [DONE] ${type} metrics"
done

cd "$PROJECT_DIR"

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Results (float compositing A/B) ==="

CSV_FILE="${OUTPUT_BASE}/metrics_summary.csv"
echo "step,dataset,type,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD" > "$CSV_FILE"

printf "%-10s %-10s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Type" "Dataset" "FID" "SSIM" "FVD" "CSIM" "Sync-C" "Sync-D" "LMD"
echo "------------------------------------------------------------------------------------"

for type in composited aligned; do
    if [ "$type" = "composited" ]; then
        log="${OUTPUT_BASE}/metrics/step-${STEP}/${DS}/metrics.log"
    else
        log="${OUTPUT_BASE}/metrics_aligned/step-${STEP}/${DS}/metrics.log"
    fi
    [ -f "$log" ] || continue
    fid=$(grep -oP 'mean_fid:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    ssim=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    fvd=$(grep -oP 'FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    csim=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    syncc=$(grep -oP 'Sync-C.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    syncd=$(grep -oP 'Sync-D.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    lmd=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    printf "%-10s %-10s %8s %8s %8s %8s %8s %8s %8s\n" \
        "$type" "$DS" "${fid:--}" "${ssim:--}" "${fvd:--}" "${csim:--}" "${syncc:--}" "${syncd:--}" "${lmd:--}"
    echo "step-${STEP},${DS},${type},${fid},${ssim},${fvd},${csim},${syncc},${syncd},${lmd}" >> "$CSV_FILE"
done

echo ""
echo "CSV: ${CSV_FILE}"
echo "Output: ${OUTPUT_BASE}/"
echo "Done."
