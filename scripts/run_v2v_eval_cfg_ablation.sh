#!/bin/bash
# CFG ablation: compare guidance_scale={1.0, 3.0} for maskall and refseq_new checkpoints.
# Existing CFG=4.5 results are in v2v_eval_maskall/ and v2v_eval_refseq_new/.
# Datasets: hdtf and hallo3 only (no mixed).
# GPUs: 0-3 (8 runs across 4 GPUs).
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
CFG_VALUES=(1.0 3.0)
DATASETS=(hdtf hallo3)
NUM_STEPS=50
OUTPUT_BASE="demo_out/v2v_eval_cfg_ablation"
INPUT_DIR="/tmp/v2v_eval_inputs"

# Checkpoints
MASKALL_EXP="/home/work/output_omniavatar_v2v_maskall"
MASKALL_STEP=5500
REFSEQ_EXP="/home/work/output_omniavatar_v2v_maskall_refseq_new_data_loss_weights"
REFSEQ_STEP=3000

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

# GT originals (full resolution)
GT_HDTF="/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data"
GT_HALLO3="/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data"

get_face_cache() {
    case "$1" in
        hdtf) echo "$FC_HDTF" ;;
        hallo3) echo "$FC_HALLO3" ;;
    esac
}

get_gt_dir() {
    case "$1" in
        hdtf) echo "$GT_HDTF" ;;
        hallo3) echo "$GT_HALLO3" ;;
    esac
}

# ─── Verify ───────────────────────────────────────────────────────────────────
echo "=== Verification ==="
[ -f "${MASKALL_EXP}/step-${MASKALL_STEP}.pt" ] || { echo "ERROR: maskall step-${MASKALL_STEP}.pt not found"; exit 1; }
[ -f "${REFSEQ_EXP}/step-${REFSEQ_STEP}.pt" ] || { echo "ERROR: refseq step-${REFSEQ_STEP}.pt not found"; exit 1; }
for ds in "${DATASETS[@]}"; do
    [ -f "${INPUT_DIR}/${ds}_latentsync.txt" ] || { echo "ERROR: input file missing for ${ds}"; exit 1; }
done
echo "  All OK"

# ─── Setup originals ─────────────────────────────────────────────────────────
echo ""
echo "=== Setting up originals ==="
for ds in "${DATASETS[@]}"; do
    orig_dir="${OUTPUT_BASE}/originals/${ds}"
    mkdir -p "$orig_dir"
    if [ "$(ls "$orig_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] ${ds} originals exist"
        continue
    fi
    gt_dir=$(get_gt_dir "$ds")
    for d in "$gt_dir"/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        [ "$name" = "face_cache" ] && continue
        [ -f "$d/sub_clip.mp4" ] && ln -sf "$d/sub_clip.mp4" "$orig_dir/${name}.mp4"
    done
    echo "  ${ds}: $(ls "$orig_dir"/*.mp4 | wc -l) linked"
done

# ─── Helper functions ─────────────────────────────────────────────────────────

run_inference() {
    local gpu=$1 model=$2 exp_path=$3 step=$4 cfg=$5 ds=$6
    local face_cache=$(get_face_cache "$ds")
    local out_dir="${OUTPUT_BASE}/${model}/cfg${cfg}/step-${step}/${ds}"
    local comp_dir="${out_dir}_composited"
    local input_file="${INPUT_DIR}/${ds}_latentsync.txt"

    if [ -d "$comp_dir" ] && [ "$(ls "$comp_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [GPU $gpu] [SKIP] ${model}/cfg${cfg}/${ds}"
        return 0
    fi

    local hp="exp_path=${exp_path},ckpt_name=step-${step}.pt,num_steps=${NUM_STEPS},guidance_scale=${cfg},output_dir=${out_dir},latentsync_inference=true,use_mouth_only_compositing=true,face_detection_cache_dir=${face_cache},mask_all_frames=true,no_first_frame_overwrite=true"

    # Add use_ref_sequence for refseq
    if [ "$model" = "refseq" ]; then
        hp="${hp},use_ref_sequence=true"
    fi

    echo "  [GPU $gpu] Inference ${model}/cfg${cfg}/${ds}..."
    mkdir -p "$out_dir"

    CUDA_VISIBLE_DEVICES=$gpu $TORCHRUN --standalone --nproc_per_node=1 \
        --master_port $((29500 + gpu)) \
        scripts/inference_v2v.py \
        --config configs/inference_v2v.yaml \
        --input_file "$input_file" \
        -hp "$hp" \
        > "${out_dir}.log" 2>&1

    # Post-process composited
    mkdir -p "$comp_dir"
    local moved=0
    for f in "$out_dir"/*_composited_audio.mp4; do
        [ -f "$f" ] || continue
        mv "$f" "$comp_dir/$(basename "$f" _composited_audio.mp4).mp4"
        moved=$((moved + 1))
    done
    echo "  [GPU $gpu] [DONE] ${model}/cfg${cfg}/${ds}: ${moved} composited"
}

run_metrics() {
    local gpu=$1 model=$2 step=$3 cfg=$4 ds=$5
    local fake_dir="${PROJECT_DIR}/${OUTPUT_BASE}/${model}/cfg${cfg}/step-${step}/${ds}_composited"
    local real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals/${ds}"
    local log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/${model}/cfg${cfg}/metrics/${ds}"

    if [ ! -d "$fake_dir" ] || [ "$(ls "$fake_dir"/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; then
        echo "  [GPU $gpu] [SKIP] metrics ${model}/cfg${cfg}/${ds} (no outputs)"
        return
    fi

    rm -rf "$log_dir"
    mkdir -p "$log_dir"
    echo "  [GPU $gpu] Metrics ${model}/cfg${cfg}/${ds}..."

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
    echo "  [GPU $gpu] [DONE] metrics ${model}/cfg${cfg}/${ds}"
}

# Per-GPU pipeline: inference both datasets, then metrics for both
run_pipeline() {
    local gpu=$1 model=$2 exp_path=$3 step=$4 cfg=$5
    for ds in "${DATASETS[@]}"; do
        run_inference "$gpu" "$model" "$exp_path" "$step" "$cfg" "$ds"
    done
    for ds in "${DATASETS[@]}"; do
        run_metrics "$gpu" "$model" "$step" "$cfg" "$ds"
    done
    echo "  [GPU $gpu] Pipeline complete: ${model}/cfg${cfg}"
}

# ─── Run: 4 combos on 4 GPUs ─────────────────────────────────────────────────
echo ""
echo "=== Running CFG ablation (4 GPUs) ==="
echo "  GPU 0: maskall/cfg1.0"
echo "  GPU 1: maskall/cfg3.0"
echo "  GPU 2: refseq/cfg1.0"
echo "  GPU 3: refseq/cfg3.0"

run_pipeline 0 maskall "$MASKALL_EXP" "$MASKALL_STEP" 1.0 &
pid0=$!
run_pipeline 1 maskall "$MASKALL_EXP" "$MASKALL_STEP" 3.0 &
pid1=$!
run_pipeline 2 refseq "$REFSEQ_EXP" "$REFSEQ_STEP" 1.0 &
pid2=$!
run_pipeline 3 refseq "$REFSEQ_EXP" "$REFSEQ_STEP" 3.0 &
pid3=$!

wait $pid0 || echo "WARNING: GPU 0 failed"
wait $pid1 || echo "WARNING: GPU 1 failed"
wait $pid2 || echo "WARNING: GPU 2 failed"
wait $pid3 || echo "WARNING: GPU 3 failed"

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== CFG Ablation Results ==="

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

CSV_FILE="${OUTPUT_BASE}/metrics_summary.csv"
echo "model,cfg,dataset,FID,SSIM,FVD,CSIM,Sync-C,Sync-D,LMD" > "$CSV_FILE"

printf "%-10s %-5s %-10s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Model" "CFG" "Dataset" "FID" "SSIM" "FVD" "CSIM" "Sync-C" "Sync-D" "LMD"
echo "--------------------------------------------------------------------------------------------"

for model in maskall refseq; do
    for cfg in "${CFG_VALUES[@]}"; do
        for ds in "${DATASETS[@]}"; do
            log="${OUTPUT_BASE}/${model}/cfg${cfg}/metrics/${ds}/metrics.log"
            extract_metrics "$log" || continue
            printf "%-10s %-5s %-10s %8s %8s %8s %8s %8s %8s %8s\n" \
                "$model" "$cfg" "$ds" "${fid:--}" "${ssim:--}" "${fvd:--}" "${csim:--}" "${syncc:--}" "${syncd:--}" "${lmd:--}"
            echo "${model},${cfg},${ds},${fid},${ssim},${fvd},${csim},${syncc},${syncd},${lmd}" >> "$CSV_FILE"
        done
    done
done

echo ""
echo "CSV: ${CSV_FILE}"
echo "Output: ${OUTPUT_BASE}/"
echo "Done."
