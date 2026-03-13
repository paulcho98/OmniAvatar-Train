#!/bin/bash
# Inference + metrics + stitching for refseq (65-channel) checkpoints.
# Uses mask_all_frames + no_first_frame_overwrite + use_ref_sequence.
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
EXP_PATH="/home/work/output_omniavatar_v2v_maskall_refseq"
STEPS=(1500)
NUM_STEPS=50
GUIDANCE_SCALE=4.5
OUTPUT_BASE="demo_out/v2v_eval_refseq"
EVAL_ALIGNED=false   # Set to true to also run metrics on 512x512 aligned outputs
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

# GT aligned (512x512) — for evaluating pre-compositing model outputs
GT_ALIGNED_HDTF="/home/work/.local/StableAvatar/latentsync_eval_hdtf/aligned_data"
GT_ALIGNED_HALLO3="/home/work/.local/StableAvatar/latentsync_eval_hallo3/aligned_data"
GT_ALIGNED_MIXED="/home/work/.local/StableAvatar/latentsync_eval_hallo3_mixed/aligned_data"

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

# ─── Setup originals (reuse if exist) ────────────────────────────────────────
echo ""
echo "=== Setting up originals ==="
for ds_info in "hdtf|${GT_HDTF}" "hallo3|${GT_HALLO3}" "hallo3_mixed|${GT_MIXED}"; do
    IFS='|' read -r ds gt_dir <<< "$ds_info"
    orig_dir="${OUTPUT_BASE}/originals/${ds}"
    mkdir -p "$orig_dir"
    if [ "$(ls "$orig_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] ${ds} originals exist"
        continue
    fi
    for d in "$gt_dir"/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        [ "$name" = "face_cache" ] && continue
        [ -f "$d/sub_clip.mp4" ] && ln -sf "$d/sub_clip.mp4" "$orig_dir/${name}.mp4"
    done
    echo "  ${ds}: $(ls "$orig_dir"/*.mp4 | wc -l) originals linked"
done

# ─── Setup aligned originals (512x512 GT, reuse if exist) ──────────────────
echo ""
echo "=== Setting up aligned originals (512x512) ==="
for ds_info in "hdtf|${GT_ALIGNED_HDTF}" "hallo3|${GT_ALIGNED_HALLO3}" "hallo3_mixed|${GT_ALIGNED_MIXED}"; do
    IFS='|' read -r ds gt_aligned_dir <<< "$ds_info"
    orig_aligned="${OUTPUT_BASE}/originals_aligned/${ds}"
    mkdir -p "$orig_aligned"
    if [ "$(ls "$orig_aligned"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] ${ds} aligned originals exist"
        continue
    fi
    for d in "$gt_aligned_dir"/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        [ "$name" = "face_cache" ] && continue
        [ -f "$d/sub_clip.mp4" ] && ln -sf "$d/sub_clip.mp4" "$orig_aligned/${name}.mp4"
    done
    echo "  ${ds}: $(ls "$orig_aligned"/*.mp4 | wc -l) aligned originals linked"
done

# ─── Helper: run one inference job ────────────────────────────────────────────
run_inference() {
    local gpu=$1 step=$2 ds=$3 face_cache=$4
    local out_dir="${OUTPUT_BASE}/step-${step}/${ds}"
    local input_file="${INPUT_DIR}/${ds}_latentsync.txt"
    local comp_dir="${out_dir}_composited"

    local aligned_dir="${out_dir}_aligned"
    if [ -d "$comp_dir" ] && [ "$(ls "$comp_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ] \
       && [ -d "$aligned_dir" ] && [ "$(ls "$aligned_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] step-${step}/${ds} (composited + aligned exist)"
        return 0
    fi

    local hp="exp_path=${EXP_PATH},ckpt_name=step-${step}.pt,num_steps=${NUM_STEPS},guidance_scale=${GUIDANCE_SCALE},output_dir=${out_dir},latentsync_inference=true,use_mouth_only_compositing=true,face_detection_cache_dir=${face_cache},mask_all_frames=true,no_first_frame_overwrite=true,use_ref_sequence=true"

    echo "  [GPU $gpu] step-${step}/${ds}..."
    mkdir -p "$out_dir"

    CUDA_VISIBLE_DEVICES=$gpu $TORCHRUN --standalone --nproc_per_node=1 \
        --master_port $((29500 + gpu)) \
        scripts/inference_v2v.py \
        --config configs/inference_v2v.yaml \
        --input_file "$input_file" \
        -hp "$hp" \
        > "${out_dir}.log" 2>&1

    # Post-process: organize composited outputs
    mkdir -p "$comp_dir"
    local moved=0
    for f in "$out_dir"/*_composited_audio.mp4; do
        [ -f "$f" ] || continue
        stem=$(basename "$f" _composited_audio.mp4)
        mv "$f" "$comp_dir/${stem}.mp4"
        moved=$((moved + 1))
    done

    # Post-process: organize aligned (512x512) outputs
    mkdir -p "$aligned_dir"
    local aligned_moved=0
    for f in "$out_dir"/*_aligned_audio.mp4; do
        [ -f "$f" ] || continue
        stem=$(basename "$f" _aligned_audio.mp4)
        mv "$f" "$aligned_dir/${stem}.mp4"
        aligned_moved=$((aligned_moved + 1))
    done
    echo "  [DONE] step-${step}/${ds}: ${moved} composited, ${aligned_moved} aligned"
}

# ─── Inference ────────────────────────────────────────────────────────────────
echo ""
echo "=== Inference ==="

if [ "${#STEPS[@]}" -eq 1 ]; then
    run_inference 0 "${STEPS[0]}" hdtf "$FC_HDTF" &
    pid0=$!
    run_inference 1 "${STEPS[0]}" hallo3 "$FC_HALLO3" &
    pid1=$!
    run_inference 2 "${STEPS[0]}" hallo3_mixed "$FC_MIXED" &
    pid2=$!

    echo "  GPU 0 (PID $pid0): step-${STEPS[0]}/hdtf"
    echo "  GPU 1 (PID $pid1): step-${STEPS[0]}/hallo3"
    echo "  GPU 2 (PID $pid2): step-${STEPS[0]}/hallo3_mixed"

    wait $pid0 || echo "WARNING: GPU 0 failed"
    wait $pid1 || echo "WARNING: GPU 1 failed"
    wait $pid2 || echo "WARNING: GPU 2 failed"
else
    echo "--- Phase 1: hdtf + hallo3 on 4 GPUs ---"
    run_inference 0 "${STEPS[0]}" hdtf "$FC_HDTF" &
    pid0=$!
    run_inference 1 "${STEPS[0]}" hallo3 "$FC_HALLO3" &
    pid1=$!
    run_inference 2 "${STEPS[1]}" hdtf "$FC_HDTF" &
    pid2=$!
    run_inference 3 "${STEPS[1]}" hallo3 "$FC_HALLO3" &
    pid3=$!

    echo "  GPU 0 (PID $pid0): step-${STEPS[0]}/hdtf"
    echo "  GPU 1 (PID $pid1): step-${STEPS[0]}/hallo3"
    echo "  GPU 2 (PID $pid2): step-${STEPS[1]}/hdtf"
    echo "  GPU 3 (PID $pid3): step-${STEPS[1]}/hallo3"

    wait $pid0 || echo "WARNING: GPU 0 failed"
    wait $pid1 || echo "WARNING: GPU 1 failed"
    wait $pid2 || echo "WARNING: GPU 2 failed"
    wait $pid3 || echo "WARNING: GPU 3 failed"

    echo "--- Phase 2: hallo3_mixed on 2 GPUs ---"
    run_inference 0 "${STEPS[0]}" hallo3_mixed "$FC_MIXED" &
    pid0=$!
    run_inference 1 "${STEPS[1]}" hallo3_mixed "$FC_MIXED" &
    pid1=$!

    echo "  GPU 0 (PID $pid0): step-${STEPS[0]}/hallo3_mixed"
    echo "  GPU 1 (PID $pid1): step-${STEPS[1]}/hallo3_mixed"

    wait $pid0 || echo "WARNING: GPU 0 failed"
    wait $pid1 || echo "WARNING: GPU 1 failed"
fi
echo "  Inference complete."

# ─── Metrics ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Metrics ==="
cd "$METRICS_REPO"

DATASETS=(hdtf hallo3 hallo3_mixed)
METRIC_JOBS=()
for step in "${STEPS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        METRIC_JOBS+=("${step}|${ds}")
    done
done

GPUS=(0 1 2 3)
idx=0
total=${#METRIC_JOBS[@]}

while [ $idx -lt $total ]; do
    pids=()
    for gpu in "${GPUS[@]}"; do
        [ $idx -ge $total ] && break
        IFS='|' read -r step ds <<< "${METRIC_JOBS[$idx]}"
        comp_dir="${PROJECT_DIR}/${OUTPUT_BASE}/step-${step}/${ds}_composited"
        real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals/${ds}"
        log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics/step-${step}/${ds}"

        if [ ! -d "$comp_dir" ] || [ "$(ls "$comp_dir"/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; then
            echo "  [SKIP] metrics step-${step}/${ds} (no composited)"
            idx=$((idx + 1))
            continue
        fi

        rm -rf "$log_dir"
        mkdir -p "$log_dir"
        echo "  [GPU $gpu] metrics step-${step}/${ds}"

        CUDA_VISIBLE_DEVICES=$gpu PATH="${CONDA_METRICS}:$PATH" \
            bash eval/run_metrics.sh \
            --real_videos_dir "$real_dir" \
            --fake_videos_dir "$comp_dir" \
            --shape_predictor_path "$SHAPE_PRED" \
            --output_dir "$log_dir" --log_path "$log_dir/metrics.log" \
            --fallback_detection_confidence 0.2 --fake_videos_top_level --all \
            > "$log_dir/eval.log" 2>&1 &
        pids+=($!)
        idx=$((idx + 1))
    done

    for pid in "${pids[@]}"; do
        wait "$pid" || echo "WARNING: metrics PID $pid failed"
    done
done

cd "$PROJECT_DIR"
echo "  All metrics complete."

# ─── Aligned Metrics (conditional) ───────────────────────────────────────────
if [ "$EVAL_ALIGNED" = true ]; then
    echo ""
    echo "=== Aligned Metrics (512x512) ==="
    cd "$METRICS_REPO"

    ALIGNED_JOBS=()
    for step in "${STEPS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            ALIGNED_JOBS+=("${step}|${ds}")
        done
    done

    idx=0
    total=${#ALIGNED_JOBS[@]}
    while [ $idx -lt $total ]; do
        pids=()
        for gpu in "${GPUS[@]}"; do
            [ $idx -ge $total ] && break
            IFS='|' read -r step ds <<< "${ALIGNED_JOBS[$idx]}"
            aligned_dir="${PROJECT_DIR}/${OUTPUT_BASE}/step-${step}/${ds}_aligned"
            real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals_aligned/${ds}"
            log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics_aligned/step-${step}/${ds}"

            if [ ! -d "$aligned_dir" ] || [ "$(ls "$aligned_dir"/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; then
                echo "  [SKIP] aligned metrics step-${step}/${ds} (no aligned outputs)"
                idx=$((idx + 1))
                continue
            fi

            rm -rf "$log_dir"
            mkdir -p "$log_dir"
            echo "  [GPU $gpu] aligned metrics step-${step}/${ds}"

            CUDA_VISIBLE_DEVICES=$gpu PATH="${CONDA_METRICS}:$PATH" \
                bash eval/run_metrics.sh \
                --real_videos_dir "$real_dir" \
                --fake_videos_dir "$aligned_dir" \
                --shape_predictor_path "$SHAPE_PRED" \
                --output_dir "$log_dir" --log_path "$log_dir/metrics.log" \
                --fallback_detection_confidence 0.2 --fake_videos_top_level --all \
                > "$log_dir/eval.log" 2>&1 &
            pids+=($!)
            idx=$((idx + 1))
        done

        for pid in "${pids[@]}"; do
            wait "$pid" || echo "WARNING: aligned metrics PID $pid failed"
        done
    done
    cd "$PROJECT_DIR"
    echo "  Aligned metrics complete."
fi

# ─── Stitching ────────────────────────────────────────────────────────────────
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

for step in "${STEPS[@]}"; do
    for ds in hdtf hallo3; do
        comp_dir="${OUTPUT_BASE}/step-${step}/${ds}_composited"
        stitch_dir="${OUTPUT_BASE}/stitched/step-${step}/${ds}"
        case "$ds" in
            hdtf) gt_base="$GT_HDTF" ;;
            hallo3) gt_base="$GT_HALLO3" ;;
        esac

        [ -d "$comp_dir" ] || continue
        echo "  step-${step}/${ds}..."
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
        echo "    ${count} stitched"
    done
done

# ─── Summary (all steps including prior runs) ────────────────────────────────
echo ""
echo "=== Metrics Summary (refseq, all checkpoints) ==="

ALL_STEPS=()
for d in "${OUTPUT_BASE}"/metrics/step-*/; do
    [ -d "$d" ] || continue
    step_name=$(basename "$d")
    ALL_STEPS+=("${step_name#step-}")
done
IFS=$'\n' ALL_STEPS=($(sort -n <<<"${ALL_STEPS[*]}")); unset IFS

printf "%-10s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Step" "Dataset" "FID" "SSIM" "FVD" "CSIM" "Sync-C" "Sync-D" "LMD"
echo "------------------------------------------------------------------------------------"

CSV_FILE="${OUTPUT_BASE}/metrics_summary.csv"
echo "step,dataset,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD" > "$CSV_FILE"

for step in "${ALL_STEPS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        log="${OUTPUT_BASE}/metrics/step-${step}/${ds}/metrics.log"
        [ -f "$log" ] || continue
        fvd=$(grep -oP 'FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
        fid=$(grep -oP 'mean_fid:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
        csim=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
        ssim=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
        lmd=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
        syncd=$(grep -oP 'Sync-D.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
        syncc=$(grep -oP 'Sync-C.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
        printf "%-10s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
            "step-${step}" "$ds" "${fid:--}" "${ssim:--}" "${fvd:--}" "${csim:--}" "${syncc:--}" "${syncd:--}" "${lmd:--}"
        echo "step-${step},${ds},${fid},${ssim},${fvd},${csim},${syncc},${syncd},${lmd}" >> "$CSV_FILE"
    done
done
echo ""
echo "CSV saved: ${CSV_FILE}"

# ─── Aligned Metrics Summary (conditional) ───────────────────────────────────
if [ "$EVAL_ALIGNED" = true ]; then
    echo ""
    echo "=== Aligned Metrics Summary (512x512, all checkpoints) ==="
    printf "%-10s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
        "Step" "Dataset" "FID" "SSIM" "FVD" "CSIM" "Sync-C" "Sync-D" "LMD"
    echo "------------------------------------------------------------------------------------"

    ALIGNED_CSV="${OUTPUT_BASE}/metrics_aligned_summary.csv"
    echo "step,dataset,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD" > "$ALIGNED_CSV"

    for step in "${ALL_STEPS[@]}"; do
        for ds in "${DATASETS[@]}"; do
            log="${OUTPUT_BASE}/metrics_aligned/step-${step}/${ds}/metrics.log"
            [ -f "$log" ] || continue
            fvd=$(grep -oP 'FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
            fid=$(grep -oP 'mean_fid:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
            csim=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
            ssim=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
            lmd=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
            syncd=$(grep -oP 'Sync-D.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
            syncc=$(grep -oP 'Sync-C.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "")
            printf "%-10s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
                "step-${step}" "$ds" "${fid:--}" "${ssim:--}" "${fvd:--}" "${csim:--}" "${syncc:--}" "${syncd:--}" "${lmd:--}"
            echo "step-${step},${ds},${fid},${ssim},${fvd},${csim},${syncc},${syncd},${lmd}" >> "$ALIGNED_CSV"
        done
    done
    echo ""
    echo "Aligned CSV saved: ${ALIGNED_CSV}"
fi

echo ""
echo "Output: ${OUTPUT_BASE}/"
echo "Done."
