#!/bin/bash
# Batch V2V inference across multiple checkpoints and eval datasets.
# Runs 9 jobs (3 checkpoints × 3 datasets) parallelized across 4 GPUs.
# Auto-resumes training when all jobs complete.
#
# Usage: bash scripts/run_v2v_eval.sh

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
EXP_PATH="/home/work/output_omniavatar_v2v_auxloss"
STEPS="5000 4500 2500"
NUM_STEPS=50
OUTPUT_BASE="demo_out/v2v_eval"
INPUT_DIR="/tmp/v2v_eval_inputs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Dataset definitions: name|path
DATASETS=(
    "hallo3|/home/work/.local/StableAvatar/latentsync_eval_hallo3/aligned_data"
    "hallo3_mixed|/home/work/.local/StableAvatar/validation_hallo3_mixed/shared_data"
    "hdtf|/home/work/.local/StableAvatar/latentsync_eval_hdtf/aligned_data"
)

# ─── Generate input files ────────────────────────────────────────────────────
mkdir -p "$INPUT_DIR"

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name dpath <<< "$entry"
    input_file="$INPUT_DIR/${name}.txt"
    > "$input_file"  # truncate
    for dir in "$dpath"/*/; do
        dir="${dir%/}"  # strip trailing slash
        base="$(basename "$dir")"
        # Skip non-sample dirs (e.g., face_cache)
        [ -f "$dir/sub_clip.mp4" ] || continue
        prompt="$(cat "$dir/prompt.txt" 2>/dev/null || echo "a person is talking")"
        echo "${prompt}@@${dir}/sub_clip.mp4@@${dir}/audio.wav" >> "$input_file"
    done
    count=$(wc -l < "$input_file")
    echo "Generated $input_file ($count samples)"
done

# ─── Build job list ──────────────────────────────────────────────────────────
# Each job: step|dataset_name|input_file
# Ordered: step-5000 first (most interesting), then 4500, then 2500
JOBS=()
for step in $STEPS; do
    for entry in "${DATASETS[@]}"; do
        IFS='|' read -r name dpath <<< "$entry"
        JOBS+=("${step}|${name}|${INPUT_DIR}/${name}.txt")
    done
done

echo ""
echo "Total jobs: ${#JOBS[@]}"
echo "GPUs: 0 1 2 3"
echo ""

# ─── Run jobs in rounds of 4 ────────────────────────────────────────────────
GPUS=(0 1 2 3)
job_idx=0
total=${#JOBS[@]}

while [ $job_idx -lt $total ]; do
    # Launch up to 4 jobs in parallel
    pids=()
    for gpu in "${GPUS[@]}"; do
        [ $job_idx -ge $total ] && break

        IFS='|' read -r step name input_file <<< "${JOBS[$job_idx]}"
        out_dir="${OUTPUT_BASE}/step-${step}/${name}"

        echo "[GPU $gpu] step-${step} × ${name} → ${out_dir}"
        mkdir -p "$out_dir"

        CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nproc_per_node=1 \
            --master_port $((29500 + gpu)) \
            scripts/inference_v2v.py \
            --config configs/inference_v2v.yaml \
            --input_file "$input_file" \
            -hp "exp_path=${EXP_PATH},ckpt_name=step-${step}.pt,num_steps=${NUM_STEPS},output_dir=${out_dir}" \
            > "${out_dir}.log" 2>&1 &
        pids+=($!)

        job_idx=$((job_idx + 1))
    done

    # Wait for this round to finish
    echo "Waiting for round (${#pids[@]} jobs)..."
    for pid in "${pids[@]}"; do
        wait "$pid" || echo "WARNING: job PID $pid exited with non-zero status"
    done
    echo "Round complete."
    echo ""
done

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "=== Eval Summary ==="
for step in $STEPS; do
    for entry in "${DATASETS[@]}"; do
        IFS='|' read -r name dpath <<< "$entry"
        out_dir="${OUTPUT_BASE}/step-${step}/${name}"
        count=$(find "$out_dir" -name "*_audio.mp4" 2>/dev/null | wc -l)
        echo "  step-${step}/${name}: ${count} videos"
    done
done

# ─── Auto-resume training ───────────────────────────────────────────────────
echo ""
echo "All inference jobs complete. Resuming training..."
bash scripts/resume_v2v_auxloss_4gpus.sh
