#!/bin/bash
# TalkVid inference + composited metrics + stitching on GPUs 2+3.
# No preprocessing needed — inference encodes audio/video live.
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
GPUS=(2 3)
EXP_PATH="/home/work/output_omniavatar_v2v_maskall_refseq_new_data_loss_weights"
STEP=3000
NUM_STEPS=50
GUIDANCE_SCALE=4.5
OUTPUT_BASE="demo_out/v2v_eval_talkvid"
DS="talkvid"

TALKVID_DIR="/home/work/.local/TalkVid/TalkVid_Data/results/top30_english_distinct_id_25fps81f_a16k/processed"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TORCHRUN="/home/work/.local/miniconda3/envs/omniavatar/bin/torchrun"
CONDA_METRICS="/home/work/.local/miniconda3/envs/latentsync-metrics/bin"
METRICS_REPO="/home/work/.local/latentsync-metrics"
SHAPE_PRED="${METRICS_REPO}/shape_predictor_68_face_landmarks.dat"

cd "$PROJECT_DIR"

# ─── Verify ───────────────────────────────────────────────────────────────────
echo "=== Verification ==="
[ -f "${EXP_PATH}/step-${STEP}.pt" ] || { echo "ERROR: step-${STEP}.pt not found"; exit 1; }
[ -d "$TALKVID_DIR" ] || { echo "ERROR: TalkVid dir not found"; exit 1; }
NUM_VIDEOS=$(ls "$TALKVID_DIR"/*.mp4 2>/dev/null | wc -l)
echo "  Checkpoint OK, ${NUM_VIDEOS} TalkVid videos found"

# ─── Generate input files ────────────────────────────────────────────────────
echo ""
echo "=== Generating input files ==="
mkdir -p "${OUTPUT_BASE}"

# Full input file (all videos)
INPUT_ALL="${OUTPUT_BASE}/${DS}_all.txt"
> "$INPUT_ALL"
for f in "$TALKVID_DIR"/*.mp4; do
    echo "a person is talking@@${f}@@${f}" >> "$INPUT_ALL"
done
echo "  ${DS}_all.txt: $(wc -l < "$INPUT_ALL") samples"

# Split for 2 GPUs
TOTAL=$(wc -l < "$INPUT_ALL")
HALF=$(( (TOTAL + 1) / 2 ))
INPUT_GPU0="${OUTPUT_BASE}/${DS}_gpu${GPUS[0]}.txt"
INPUT_GPU1="${OUTPUT_BASE}/${DS}_gpu${GPUS[1]}.txt"
head -n "$HALF" "$INPUT_ALL" > "$INPUT_GPU0"
tail -n +$((HALF + 1)) "$INPUT_ALL" > "$INPUT_GPU1"
echo "  GPU ${GPUS[0]}: $(wc -l < "$INPUT_GPU0") samples"
echo "  GPU ${GPUS[1]}: $(wc -l < "$INPUT_GPU1") samples"

# ─── Setup originals (copy, not symlink — metrics uses find -type f) ─────────
echo ""
echo "=== Setting up originals ==="
ORIG_DIR="${OUTPUT_BASE}/originals/${DS}"
mkdir -p "$ORIG_DIR"
if [ "$(ls "$ORIG_DIR"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  [SKIP] originals exist ($(ls "$ORIG_DIR"/*.mp4 | wc -l) files)"
else
    for f in "$TALKVID_DIR"/*.mp4; do
        cp "$f" "$ORIG_DIR/$(basename "$f")"
    done
    echo "  Copied ${NUM_VIDEOS} originals"
fi

# ─── Helper functions ─────────────────────────────────────────────────────────

run_inference() {
    local gpu=$1 input_file=$2
    local out_dir="${OUTPUT_BASE}/step-${STEP}/${DS}_gpu${gpu}"
    local comp_dir="${OUTPUT_BASE}/step-${STEP}/${DS}_composited"
    local aligned_dir="${OUTPUT_BASE}/step-${STEP}/${DS}_aligned"
    local fc="${OUTPUT_BASE}/face_cache"

    local n_input=$(wc -l < "$input_file")
    [ "$n_input" -eq 0 ] && { echo "  [GPU $gpu] No samples, skipping"; return 0; }

    echo "  [GPU $gpu] Inference (${n_input} samples)..."
    mkdir -p "$out_dir"

    local hp="exp_path=${EXP_PATH},ckpt_name=step-${STEP}.pt,num_steps=${NUM_STEPS},guidance_scale=${GUIDANCE_SCALE},output_dir=${out_dir},latentsync_inference=true,use_mouth_only_compositing=true,face_detection_cache_dir=${fc},mask_all_frames=true,no_first_frame_overwrite=true,use_ref_sequence=true"

    CUDA_VISIBLE_DEVICES=$gpu $TORCHRUN --standalone --nproc_per_node=1 \
        --master_port $((29500 + gpu)) \
        scripts/inference_v2v.py \
        --config configs/inference_v2v.yaml \
        --input_file "$input_file" \
        -hp "$hp" \
        > "${out_dir}.log" 2>&1

    # Move outputs to shared composited/aligned dirs
    mkdir -p "$comp_dir" "$aligned_dir"
    local moved=0 aligned_moved=0
    for f in "$out_dir"/*_composited_audio.mp4; do
        [ -f "$f" ] || continue
        mv "$f" "$comp_dir/$(basename "$f" _composited_audio.mp4).mp4"
        moved=$((moved + 1))
    done
    for f in "$out_dir"/*_aligned_audio.mp4; do
        [ -f "$f" ] || continue
        mv "$f" "$aligned_dir/$(basename "$f" _aligned_audio.mp4).mp4"
        aligned_moved=$((aligned_moved + 1))
    done
    echo "  [GPU $gpu] [DONE] ${moved} composited, ${aligned_moved} aligned"
}

run_metrics() {
    local gpu=$1
    local fake_dir="${PROJECT_DIR}/${OUTPUT_BASE}/step-${STEP}/${DS}_composited"
    local real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals/${DS}"
    local log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics/step-${STEP}/${DS}"

    if [ ! -d "$fake_dir" ] || [ "$(ls "$fake_dir"/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "  [SKIP] No composited outputs for metrics"
        return
    fi

    rm -rf "$log_dir"
    mkdir -p "$log_dir"
    echo "  [GPU $gpu] Running composited metrics..."

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
    echo "  [GPU $gpu] [DONE] Composited metrics"
}

run_stitch() {
    local comp_dir="${OUTPUT_BASE}/step-${STEP}/${DS}_composited"
    local stitch_dir="${OUTPUT_BASE}/stitched/${DS}"
    [ -d "$comp_dir" ] || return 0

    echo "  Stitching 2-way GT | Generated..."
    mkdir -p "$stitch_dir"
    for gen_video in "$comp_dir"/*.mp4; do
        local name=$(basename "$gen_video")
        local gt_video="${OUTPUT_BASE}/originals/${DS}/${name}"
        local out_video="${stitch_dir}/${name}"
        [ -f "$gt_video" ] || continue
        [ -f "$out_video" ] && continue

        # Get frame counts for proper trimming
        local gt_frames=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$gt_video")
        local gen_frames=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$gen_video")
        local min_frames=$((gt_frames < gen_frames ? gt_frames : gen_frames))

        local gt_h=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$gt_video")
        gt_h=$(( (gt_h / 2) * 2 ))

        ffmpeg -i "$gt_video" -i "$gen_video" \
            -filter_complex \
            "[0:v]trim=end_frame=${min_frames},setpts=PTS-STARTPTS,scale=-2:${gt_h}[gt];
             [1:v]trim=end_frame=${min_frames},setpts=PTS-STARTPTS,scale=-2:${gt_h}[gen];
             [gt][gen]hstack=inputs=2" \
            -c:v libx264 -preset fast -crf 18 -c:a aac -map 1:a? -shortest -y \
            "$out_video" </dev/null 2>/dev/null
    done
    echo "  [DONE] $(ls "$stitch_dir"/*.mp4 2>/dev/null | wc -l) stitched videos"
}

# ─── Run pipeline ────────────────────────────────────────────────────────────
echo ""
echo "=== Inference (GPUs ${GPUS[*]}) ==="

# Check if inference already done
COMP_DIR="${OUTPUT_BASE}/step-${STEP}/${DS}_composited"
if [ -d "$COMP_DIR" ] && [ "$(ls "$COMP_DIR"/*.mp4 2>/dev/null | wc -l)" -ge "$NUM_VIDEOS" ]; then
    echo "  [SKIP] Inference already complete ($(ls "$COMP_DIR"/*.mp4 | wc -l) composited)"
else
    run_inference "${GPUS[0]}" "$INPUT_GPU0" &
    pid0=$!
    run_inference "${GPUS[1]}" "$INPUT_GPU1" &
    pid1=$!
    echo "  GPU ${GPUS[0]} (PID $pid0): samples 1-${HALF}"
    echo "  GPU ${GPUS[1]} (PID $pid1): samples $((HALF+1))-${TOTAL}"
    wait $pid0 || echo "WARNING: GPU ${GPUS[0]} failed"
    wait $pid1 || echo "WARNING: GPU ${GPUS[1]} failed"
fi

echo ""
echo "=== Metrics ==="
run_metrics "${GPUS[0]}"

echo ""
echo "=== Stitching ==="
run_stitch

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Results ==="

CSV_FILE="${OUTPUT_BASE}/metrics_summary.csv"
echo "step,dataset,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD" > "$CSV_FILE"

log="${OUTPUT_BASE}/metrics/step-${STEP}/${DS}/metrics.log"
if [ -f "$log" ]; then
    fid=$(grep -oP 'mean_fid:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    ssim=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    fvd=$(grep -oP 'FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    csim=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    syncc=$(grep -oP 'Sync-C.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    syncd=$(grep -oP 'Sync-D.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
    lmd=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")

    printf "%-10s %-10s %8s %8s %8s %8s %8s %8s %8s\n" \
        "Step" "Dataset" "FID" "SSIM" "FVD" "CSIM" "Sync-C" "Sync-D" "LMD"
    echo "------------------------------------------------------------------------------------"
    printf "%-10s %-10s %8s %8s %8s %8s %8s %8s %8s\n" \
        "step-${STEP}" "$DS" "${fid:--}" "${ssim:--}" "${fvd:--}" "${csim:--}" "${syncc:--}" "${syncd:--}" "${lmd:--}"
    echo "step-${STEP},${DS},${fid},${ssim},${fvd},${csim},${syncc},${syncd},${lmd}" >> "$CSV_FILE"
else
    echo "  No metrics log found"
fi

echo ""
echo "CSV: ${CSV_FILE}"
echo "Output: ${OUTPUT_BASE}/"
echo "Composited: ${OUTPUT_BASE}/step-${STEP}/${DS}_composited/"
echo "Aligned: ${OUTPUT_BASE}/step-${STEP}/${DS}_aligned/"
echo "Stitched: ${OUTPUT_BASE}/stitched/${DS}/"
echo "Done."
