#!/bin/bash
# Balanced follow-up: runs remaining 11 jobs after round 1 of the main script.
# Each GPU runs its assigned jobs sequentially. All 4 GPUs run in parallel.
# Then: post-processing → metrics → training.
#
# Usage: bash scripts/run_v2v_eval_ablation_round2.sh
set -euo pipefail

# ─── Configuration (same as main script) ─────────────────────────────────────
EXP_PATH="/home/work/output_omniavatar_v2v_auxloss"
NUM_STEPS=50
OUTPUT_BASE="demo_out/v2v_eval_ablation"
INPUT_DIR="/tmp/v2v_eval_inputs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TORCHRUN="/home/work/.local/miniconda3/envs/omniavatar/bin/torchrun"
PYTHON="/home/work/.local/miniconda3/envs/omniavatar/bin/python"
CONDA_METRICS="/home/work/.local/miniconda3/envs/latentsync-metrics/bin"
METRICS_REPO="/home/work/.local/latentsync-metrics"
SHAPE_PRED="${METRICS_REPO}/shape_predictor_68_face_landmarks.dat"

cd "$PROJECT_DIR"

# Face cache dirs
FC_HDTF="/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data/face_cache"
FC_HALLO3="/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data/face_cache"
FC_MIXED="/home/work/.local/StableAvatar/validation_hallo3_mixed/shared_data/face_cache"

# ─── Helper: run one inference job ───────────────────────────────────────────
run_job() {
    local gpu=$1 label=$2 step=$3 cfg=$4 noffo=$5 ds_name=$6 face_cache=$7
    local out_dir="${OUTPUT_BASE}/${label}/${ds_name}"
    local input_file="${INPUT_DIR}/${ds_name}_latentsync.txt"
    local comp_dir="${out_dir}_composited"

    # Skip if already done
    if [ -d "$comp_dir" ] && [ "$(ls "$comp_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] ${label}/${ds_name} (composited exists)"
        return 0
    fi
    if [ -d "$out_dir" ]; then
        local existing=$(find "$out_dir" -name "*_composited_audio.mp4" 2>/dev/null | wc -l)
        if [ "$existing" -gt 0 ]; then
            echo "  [SKIP-INF] ${label}/${ds_name} ($existing composited videos, needs post-processing)"
            # Do post-processing only
            mkdir -p "$comp_dir"
            for f in "$out_dir"/*_composited_audio.mp4; do
                [ -f "$f" ] || continue
                stem=$(basename "$f" _composited_audio.mp4)
                mv "$f" "$comp_dir/${stem}.mp4"
            done
            return 0
        fi
    fi

    # Build -hp string
    local hp="exp_path=${EXP_PATH},ckpt_name=step-${step}.pt,num_steps=${NUM_STEPS},guidance_scale=${cfg},output_dir=${out_dir},latentsync_inference=true,use_mouth_only_compositing=true,face_detection_cache_dir=${face_cache}"
    if [ "$noffo" = "true" ]; then
        hp="${hp},no_first_frame_overwrite=true"
    fi

    echo "  [GPU $gpu] ${label}/${ds_name} (cfg=${cfg}, noffo=${noffo})"
    mkdir -p "$out_dir"

    CUDA_VISIBLE_DEVICES=$gpu $TORCHRUN --standalone --nproc_per_node=1 \
        --master_port $((29500 + gpu)) \
        scripts/inference_v2v.py \
        --config configs/inference_v2v.yaml \
        --input_file "$input_file" \
        -hp "$hp" \
        > "${out_dir}.log" 2>&1

    # Post-process: move composited videos to clean dir
    mkdir -p "$comp_dir"
    local moved=0
    for f in "$out_dir"/*_composited_audio.mp4; do
        [ -f "$f" ] || continue
        stem=$(basename "$f" _composited_audio.mp4)
        mv "$f" "$comp_dir/${stem}.mp4"
        moved=$((moved + 1))
    done
    echo "  [DONE] ${label}/${ds_name}: $moved videos composited"
}

# ─── Balanced GPU assignment (11 remaining jobs) ─────────────────────────────
echo "=== Balanced inference: 11 jobs across 4 GPUs ==="

# GPU 0: ~206 min
(
    run_job 0 5500_noffo 5500 4.5 true  hdtf         "$FC_HDTF"
    run_job 0 5500_nocfg 5500 1.0 false hallo3       "$FC_HALLO3"
    run_job 0 5500_nocfg 5500 1.0 false hallo3_mixed "$FC_MIXED"
) &
pid0=$!

# GPU 1: ~210 min
(
    run_job 1 5500_cfg   5500 4.5 false hdtf         "$FC_HDTF"
    run_job 1 2500_noffo 2500 4.5 true  hallo3_mixed "$FC_MIXED"
    run_job 1 5500_noffo 5500 4.5 true  hallo3_mixed "$FC_MIXED"
) &
pid1=$!

# GPU 2: ~221 min
(
    run_job 2 2500_noffo 2500 4.5 true  hallo3       "$FC_HALLO3"
    run_job 2 5500_nocfg 5500 1.0 false hdtf         "$FC_HDTF"
    run_job 2 5500_cfg   5500 4.5 false hallo3_mixed "$FC_MIXED"
) &
pid2=$!

# GPU 3: ~222 min
(
    run_job 3 5500_cfg   5500 4.5 false hallo3       "$FC_HALLO3"
    run_job 3 5500_noffo 5500 4.5 true  hallo3       "$FC_HALLO3"
) &
pid3=$!

echo "  GPU 0 (PID $pid0): noffo/hdtf → nocfg/hallo3 → nocfg/mixed (~206 min)"
echo "  GPU 1 (PID $pid1): cfg/hdtf → noffo/mixed → noffo/mixed (~210 min)"
echo "  GPU 2 (PID $pid2): noffo/hallo3 → nocfg/hdtf → cfg/mixed (~221 min)"
echo "  GPU 3 (PID $pid3): cfg/hallo3 → noffo/hallo3 (~222 min)"
echo ""

wait $pid0 || echo "WARNING: GPU 0 had failures"
wait $pid1 || echo "WARNING: GPU 1 had failures"
wait $pid2 || echo "WARNING: GPU 2 had failures"
wait $pid3 || echo "WARNING: GPU 3 had failures"

echo "  All inference complete."

# ─── Metrics ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Metrics ==="
cd "$METRICS_REPO"

COMBOS=("2500_nocfg" "2500_noffo" "5500_nocfg" "5500_noffo" "5500_cfg")
DS_NAMES=("hdtf" "hallo3" "hallo3_mixed")
GPUS=(0 1 2 3)

METRIC_JOBS=()
for label in "${COMBOS[@]}"; do
    for ds_name in "${DS_NAMES[@]}"; do
        comp_dir="${PROJECT_DIR}/${OUTPUT_BASE}/${label}/${ds_name}_composited"
        log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics/${label}/${ds_name}"
        real_dir="${PROJECT_DIR}/${OUTPUT_BASE}/originals/${ds_name}"
        METRIC_JOBS+=("${label}|${ds_name}|${comp_dir}|${real_dir}|${log_dir}")
    done
done

metric_idx=0
metric_total=${#METRIC_JOBS[@]}

while [ $metric_idx -lt $metric_total ]; do
    pids=()
    for gpu in "${GPUS[@]}"; do
        [ $metric_idx -ge $metric_total ] && break

        IFS='|' read -r label ds_name comp_dir real_dir log_dir <<< "${METRIC_JOBS[$metric_idx]}"

        if [ -f "$log_dir/metrics.log" ] && ! grep -q "failures" "$log_dir/metrics.log"; then
            echo "  [SKIP] metrics ${label}/${ds_name}"
            metric_idx=$((metric_idx + 1))
            continue
        fi

        if [ ! -d "$comp_dir" ] || [ "$(ls "$comp_dir"/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; then
            echo "  [SKIP] metrics ${label}/${ds_name} (no composited videos)"
            metric_idx=$((metric_idx + 1))
            continue
        fi

        mkdir -p "$log_dir"
        echo "  [GPU $gpu] metrics ${label}/${ds_name}"

        CUDA_VISIBLE_DEVICES=$gpu PATH="${CONDA_METRICS}:$PATH" \
            bash eval/run_metrics.sh \
            --real_videos_dir "$real_dir" \
            --fake_videos_dir "$comp_dir" \
            --shape_predictor_path "$SHAPE_PRED" \
            --output_dir "$log_dir" --log_path "$log_dir/metrics.log" \
            --fallback_detection_confidence 0.2 --fake_videos_top_level --all \
            > "$log_dir/eval.log" 2>&1 &
        pids+=($!)
        metric_idx=$((metric_idx + 1))
    done

    for pid in "${pids[@]}"; do
        wait "$pid" || echo "WARNING: metrics PID $pid failed"
    done
done

cd "$PROJECT_DIR"
echo "  All metrics complete."

# ─── Summary Table ───────────────────────────────────────────────────────────
echo ""
echo "=== Ablation Metrics Summary ==="
printf "%-14s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
    "Combo" "Dataset" "FVD" "FID" "CSIM" "SSIM" "LMD" "Sync-D" "Sync-C"
echo "------------------------------------------------------------------------------------"
for label in "${COMBOS[@]}"; do
    for ds_name in "${DS_NAMES[@]}"; do
        log="${OUTPUT_BASE}/metrics/${label}/${ds_name}/metrics.log"
        [ -f "$log" ] || continue
        fvd=$(grep -oP 'FVD:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        fid=$(grep -oP 'mean_fid:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        csim=$(grep -oP 'cosine similarity:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        ssim=$(grep -oP 'mean_ssim:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        lmd=$(grep -oP 'mean_lmd:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        syncd=$(grep -oP 'Sync-D.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        syncc=$(grep -oP 'Sync-C.*:\s*\K[\d.]+' "$log" 2>/dev/null || echo "-")
        printf "%-14s %-14s %8s %8s %8s %8s %8s %8s %8s\n" \
            "$label" "$ds_name" "$fvd" "$fid" "$csim" "$ssim" "$lmd" "$syncd" "$syncc"
    done
done

# ─── Training ────────────────────────────────────────────────────────────────
echo ""
echo "=== Launching training from step-5500 ==="

SAMPLE_DIR=$(head -1 /home/work/stableavatar_data/v2v_training_data/video_square_path.txt)
if [ ! -f "${SAMPLE_DIR}/vae_latents_mask_all.pt" ]; then
    echo "ERROR: vae_latents_mask_all.pt not found in ${SAMPLE_DIR}"
    echo "Run scripts/run_precompute_vae_masked.sh first!"
    exit 1
fi
echo "  vae_latents_mask_all.pt verified."

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate_4gpus.yaml \
    scripts/train_v2v.py \
    --dit_paths "pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors" \
    --text_encoder_path pretrained_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth \
    --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
    --wav2vec_path pretrained_models/wav2vec2-base-960h \
    --omniavatar_ckpt /home/work/output_omniavatar_v2v_auxloss/step-5500.pt \
    --data_list_path /home/work/stableavatar_data/v2v_training_data/video_square_path.txt \
    --latentsync_mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
    --use_precomputed_vae \
    --use_precomputed_audio \
    --use_precomputed_text_emb \
    --num_frames 81 --height 512 --width 512 \
    --num_epochs 1000 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 2 \
    --max_grad_norm 1 \
    --weight_decay 0.01 \
    --seed 42 \
    --use_gradient_checkpointing \
    --text_drop_prob 0.1 \
    --audio_drop_prob 0.1 \
    --output_path /home/work/output_omniavatar_v2v_maskall \
    --save_steps 500 \
    --use_sync_loss \
    --use_lpips_loss \
    --use_trepa_loss \
    --use_vae_gradient_checkpointing \
    --aux_recon_weight 1.0 \
    --aux_sync_weight 0.1 \
    --aux_lpips_weight 0.1 \
    --aux_trepa_weight 10.0 \
    --aux_num_frames 21 \
    --sync_chunk_size 16 \
    --sync_chunk_stride 8 \
    --sync_num_supervised_frames 80 \
    --val_data_recon /home/work/stableavatar_data/v2v_validation_data/recon \
    --val_data_mixed /home/work/stableavatar_data/v2v_validation_data/mixed \
    --val_max_samples 4 \
    --val_num_inference_steps 25 \
    --validation_steps 500 \
    --use_wandb \
    --wandb_entity "paulhcho" \
    --wandb_project "OmniAvatar-V2V" \
    --wandb_run_name "v2v_14B_maskall_from5500" \
    --wandb_log_every 1 \
    --compute_sync_metrics \
    --offload_frozen \
    --mask_all_frames \
    --no_first_frame_overwrite \
    --wandb_api_key "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"
