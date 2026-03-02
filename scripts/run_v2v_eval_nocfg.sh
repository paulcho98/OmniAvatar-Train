#!/bin/bash
# V2V evaluation WITHOUT CFG: inference → compositing → metrics, all-in-one.
# Runs on GPUs 2+3.
#
# Usage: bash scripts/run_v2v_eval_nocfg.sh
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
EXP_PATH="/home/work/output_omniavatar_v2v_auxloss"
STEPS="5500 5000 4500 2500"
NUM_STEPS=50
GUIDANCE_SCALE=1.0
OUTPUT_BASE="demo_out/v2v_eval_nocfg"
INPUT_DIR="/tmp/v2v_eval_inputs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/work/.local/miniconda3/envs/omniavatar/bin/python"
CONDA_METRICS="/home/work/.local/miniconda3/envs/latentsync-metrics/bin"
METRICS_REPO="/home/work/.local/latentsync-metrics"
SHAPE_PRED="${METRICS_REPO}/shape_predictor_68_face_landmarks.dat"
GPUS=(2 3)

cd "$PROJECT_DIR"

# Datasets: name|aligned_path|originals_dir|face_cache_dir
DATASETS=(
    "hdtf|/home/work/.local/StableAvatar/latentsync_eval_hdtf/aligned_data|/home/work/.local/HDTF/HDTF_original_testset_81frames/videos_cfr|/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data/face_cache"
    "hallo3|/home/work/.local/StableAvatar/latentsync_eval_hallo3/aligned_data|/home/work/.local/Hallo3_validation/validation_81frames/videos_cfr|/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data/face_cache"
)

# ─── Generate input files ───────────────────────────────────────────────────
mkdir -p "$INPUT_DIR"
for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name dpath orig_dir face_cache <<< "$entry"
    input_file="$INPUT_DIR/${name}.txt"
    if [ -s "$input_file" ]; then
        echo "Reusing $input_file ($(wc -l < "$input_file") samples)"
        continue
    fi
    > "$input_file"
    for dir in "$dpath"/*/; do
        dir="${dir%/}"
        [ -f "$dir/sub_clip.mp4" ] || continue
        prompt="$(cat "$dir/prompt.txt" 2>/dev/null || echo "a person is talking")"
        echo "${prompt}@@${dir}/sub_clip.mp4@@${dir}/audio.wav" >> "$input_file"
    done
    echo "Generated $input_file ($(wc -l < "$input_file") samples)"
done

# ─── Per-step pipeline: inference → compositing → metrics ────────────────────
for step in $STEPS; do
    echo ""
    echo "================================================================"
    echo "  STEP ${step} — guidance_scale=${GUIDANCE_SCALE}"
    echo "================================================================"

    # ── Phase 1: Inference ───────────────────────────────────────────────
    echo "--- Phase 1: Inference ---"
    inf_pids=()
    gpu_idx=0
    for entry in "${DATASETS[@]}"; do
        IFS='|' read -r name dpath orig_dir face_cache <<< "$entry"
        out_dir="${OUTPUT_BASE}/step-${step}/${name}"
        input_file="$INPUT_DIR/${name}.txt"

        existing=$(find "$out_dir" -name "*_audio.mp4" 2>/dev/null | wc -l)
        if [ "$existing" -gt 0 ]; then
            echo "  [SKIP] inference step-${step}/${name} ($existing videos)"
            continue
        fi

        gpu=${GPUS[$gpu_idx]}
        echo "  [GPU $gpu] inference step-${step}/${name}"
        mkdir -p "$out_dir"

        CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nproc_per_node=1 \
            --master_port $((29500 + gpu)) \
            scripts/inference_v2v.py \
            --config configs/inference_v2v.yaml \
            --input_file "$input_file" \
            -hp "exp_path=${EXP_PATH},ckpt_name=step-${step}.pt,num_steps=${NUM_STEPS},guidance_scale=${GUIDANCE_SCALE},output_dir=${out_dir}" \
            > "${out_dir}.log" 2>&1 &
        inf_pids+=($!)
        gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
    done
    for pid in "${inf_pids[@]}"; do
        wait "$pid" || echo "WARNING: inference PID $pid failed"
    done
    echo "  Inference done."

    # ── Phase 2: Compositing ─────────────────────────────────────────────
    echo "--- Phase 2: Compositing ---"
    comp_pids=()
    gpu_idx=0
    for entry in "${DATASETS[@]}"; do
        IFS='|' read -r name dpath orig_dir face_cache <<< "$entry"
        gen_dir="${OUTPUT_BASE}/step-${step}/${name}"
        comp_dir="${OUTPUT_BASE}/step-${step}/${name}_composited"

        if [ -d "$comp_dir" ] && [ "$(ls "$comp_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "  [SKIP] compositing step-${step}/${name}"
            continue
        fi

        gpu=${GPUS[$gpu_idx]}
        echo "  [GPU $gpu] compositing step-${step}/${name}"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON scripts/composite_v2v.py \
            --generated_dir "$gen_dir" \
            --original_videos_dir "$orig_dir" \
            --output_dir "$comp_dir" \
            --face_cache_dir "${OUTPUT_BASE}/face_cache/${name}" \
            --existing_face_cache_dir "$face_cache" &
        comp_pids+=($!)
        gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
    done
    for pid in "${comp_pids[@]}"; do
        wait "$pid" || echo "WARNING: compositing PID $pid failed"
    done
    echo "  Compositing done."

    # ── Phase 3: Metrics ─────────────────────────────────────────────────
    echo "--- Phase 3: Metrics ---"
    cd "$METRICS_REPO"
    eval_pids=()
    gpu_idx=0
    for entry in "${DATASETS[@]}"; do
        IFS='|' read -r name dpath orig_dir face_cache <<< "$entry"
        comp_dir="${PROJECT_DIR}/${OUTPUT_BASE}/step-${step}/${name}_composited"
        log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics/step-${step}/${name}"
        mkdir -p "$log_dir"

        if [ -f "$log_dir/metrics.log" ] && ! grep -q "failures" "$log_dir/metrics.log"; then
            echo "  [SKIP] metrics step-${step}/${name}"
            continue
        fi

        gpu=${GPUS[$gpu_idx]}
        echo "  [GPU $gpu] metrics step-${step}/${name}"
        CUDA_VISIBLE_DEVICES=$gpu PATH="${CONDA_METRICS}:$PATH" \
            bash eval/run_metrics.sh \
            --real_videos_dir "$orig_dir" \
            --fake_videos_dir "$comp_dir" \
            --shape_predictor_path "$SHAPE_PRED" \
            --output_dir "$log_dir" --log_path "$log_dir/metrics.log" \
            --fallback_detection_confidence 0.2 --fake_videos_top_level --all \
            > "$log_dir/eval.log" 2>&1 &
        eval_pids+=($!)
        gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
    done
    for pid in "${eval_pids[@]}"; do
        wait "$pid" || echo "WARNING: metrics PID $pid failed"
    done
    cd "$PROJECT_DIR"
    echo "  Metrics done."

    # Print step results
    echo "--- Step ${step} Results ---"
    for name in hdtf hallo3; do
        log="${OUTPUT_BASE}/metrics/step-${step}/${name}/metrics.log"
        [ -f "$log" ] || continue
        fvd=$(grep -oP 'FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        syncd=$(grep -oP 'Mean SyncNet Min Distance.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        echo "  ${name}: FVD=${fvd} Sync-D=${syncd}"
    done
done

# ─── Final Summary ──────────────────────────────────────────────────────────
echo ""
echo "=== No-CFG Metrics Summary ==="
printf "%-12s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Step" "Dataset" "FVD" "FID" "CSIM" "SSIM" "LMD" "Sync-D" "Sync-C"
echo "--------------------------------------------------------------------------------"
for step in $STEPS; do
    for name in hdtf hallo3; do
        log="${OUTPUT_BASE}/metrics/step-${step}/${name}/metrics.log"
        [ -f "$log" ] || continue
        fvd=$(grep -oP 'FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        fid=$(grep -oP 'mean_fid:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        csim=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        ssim=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        lmd=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        syncd=$(grep -oP 'Mean SyncNet Min Distance.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        syncc=$(grep -oP 'Mean SyncNet Confidence.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        printf "%-12s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
            "step-${step}" "$name" "$fvd" "$fid" "$csim" "$ssim" "$lmd" "$syncd" "$syncc"
    done
done
