#!/bin/bash
# V2V ablation evaluation: 5 inference combos × 3 datasets.
# Uses latentsync_inference for inline compositing (no separate compositing step).
# Existing face caches are reused directly.
#
# Combos: 2500_nocfg, 2500_noffo, 5500_nocfg, 5500_noffo, 5500_cfg
# Datasets: hdtf (33), hallo3 (30), hallo3_mixed (12)
#
# After eval: launches training from step-5500 with --mask_all_frames --no_first_frame_overwrite.
#
# Usage: bash scripts/run_v2v_eval_ablation.sh
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
EXP_PATH="/home/work/output_omniavatar_v2v_auxloss"
NUM_STEPS=50
OUTPUT_BASE="demo_out/v2v_eval_ablation"
INPUT_DIR="/tmp/v2v_eval_inputs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/work/.local/miniconda3/envs/omniavatar/bin/python"
CONDA_METRICS="/home/work/.local/miniconda3/envs/latentsync-metrics/bin"
METRICS_REPO="/home/work/.local/latentsync-metrics"
SHAPE_PRED="${METRICS_REPO}/shape_predictor_68_face_landmarks.dat"
GPUS=(0 1 2 3)

cd "$PROJECT_DIR"

# ─── Dataset definitions ────────────────────────────────────────────────────
# name|aligned_path|originals_video_dir|face_cache_dir|metrics_originals_dir
DATASETS=(
    "hdtf|/home/work/.local/StableAvatar/latentsync_eval_hdtf/aligned_data|/home/work/.local/HDTF/HDTF_original_testset_81frames/videos_cfr|/home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data/face_cache|/home/work/.local/HDTF/HDTF_original_testset_81frames/videos_cfr"
    "hallo3|/home/work/.local/StableAvatar/latentsync_eval_hallo3/aligned_data|/home/work/.local/Hallo3_validation/validation_81frames/videos_cfr|/home/work/.local/StableAvatar/validation_hallo3/shared_hallo3_data/face_cache|/home/work/.local/Hallo3_validation/validation_81frames/videos_cfr"
    "hallo3_mixed|/home/work/.local/StableAvatar/validation_hallo3_mixed/shared_data||/home/work/.local/StableAvatar/validation_hallo3_mixed/shared_data/face_cache|"
)

# ─── Combo definitions ──────────────────────────────────────────────────────
# label|step|guidance_scale|no_first_frame_overwrite
COMBOS=(
    "2500_nocfg|2500|1.0|false"
    "2500_noffo|2500|4.5|true"
    "5500_nocfg|5500|1.0|false"
    "5500_noffo|5500|4.5|true"
    "5500_cfg|5500|4.5|false"
)

# ─── Generate input files ───────────────────────────────────────────────────
echo "=== Generating input files ==="
mkdir -p "$INPUT_DIR"

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name aligned_path orig_video_dir face_cache metrics_orig <<< "$entry"
    input_file="$INPUT_DIR/${name}_latentsync.txt"

    if [ -s "$input_file" ]; then
        echo "  Reusing $input_file ($(wc -l < "$input_file") samples)"
        continue
    fi

    > "$input_file"

    if [ "$name" = "hallo3_mixed" ]; then
        # Mixed: video + audio in same directory
        for dir in "$aligned_path"/v_*/; do
            dir="${dir%/}"
            [ -f "$dir/sub_clip.mp4" ] || continue
            prompt="$(cat "$dir/prompt.txt" 2>/dev/null || echo "a person is talking")"
            echo "${prompt}@@${dir}/sub_clip.mp4@@${dir}/audio.wav" >> "$input_file"
        done
    else
        # hdtf/hallo3: video from originals, audio+prompt from aligned data
        for dir in "$aligned_path"/*/; do
            dir="${dir%/}"
            sample="$(basename "$dir")"
            [ -f "$dir/audio.wav" ] || continue
            prompt="$(cat "$dir/prompt.txt" 2>/dev/null || echo "a person is talking")"
            # Find matching full-res original
            video=""
            for candidate in "${orig_video_dir}/${sample}_cfr25.mp4" "${orig_video_dir}/${sample}.mp4"; do
                if [ -f "$candidate" ]; then
                    video="$candidate"
                    break
                fi
            done
            if [ -z "$video" ]; then
                echo "  WARNING: no original for $sample, skipping"
                continue
            fi
            echo "${prompt}@@${video}@@${dir}/audio.wav" >> "$input_file"
        done
    fi

    echo "  Generated $input_file ($(wc -l < "$input_file") samples)"
done

# ─── Create originals symlink dir for hallo3_mixed metrics ───────────────────
MIXED_ORIG="${OUTPUT_BASE}/originals/hallo3_mixed"
if [ ! -d "$MIXED_ORIG" ] || [ "$(ls "$MIXED_ORIG"/*.mp4 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "  Creating mixed originals symlink dir: $MIXED_ORIG"
    mkdir -p "$MIXED_ORIG"
    for dir in /home/work/.local/StableAvatar/validation_hallo3_mixed/shared_data/v_*/; do
        dir="${dir%/}"
        sample="$(basename "$dir")"
        [ -f "$dir/sub_clip.mp4" ] || continue
        ln -sf "$dir/sub_clip.mp4" "$MIXED_ORIG/${sample}.mp4"
    done
    echo "  Created $(ls "$MIXED_ORIG"/*.mp4 2>/dev/null | wc -l) symlinks"
fi

# ─── Build flat job list ─────────────────────────────────────────────────────
echo ""
echo "=== Building job list ==="
JOBS=()
for combo_entry in "${COMBOS[@]}"; do
    IFS='|' read -r label step cfg noffo <<< "$combo_entry"
    for ds_entry in "${DATASETS[@]}"; do
        IFS='|' read -r ds_name aligned_path orig_video_dir face_cache metrics_orig <<< "$ds_entry"
        JOBS+=("${label}|${step}|${cfg}|${noffo}|${ds_name}|${face_cache}")
    done
done
echo "Total jobs: ${#JOBS[@]} (${#COMBOS[@]} combos × 3 datasets)"

# ─── Run inference in rounds ─────────────────────────────────────────────────
echo ""
echo "=== Phase 1: Inference ==="
job_idx=0
total=${#JOBS[@]}

while [ $job_idx -lt $total ]; do
    pids=()
    round_jobs=()

    for gpu in "${GPUS[@]}"; do
        [ $job_idx -ge $total ] && break

        IFS='|' read -r label step cfg noffo ds_name face_cache <<< "${JOBS[$job_idx]}"
        out_dir="${OUTPUT_BASE}/${label}/${ds_name}"
        input_file="${INPUT_DIR}/${ds_name}_latentsync.txt"

        # Skip if already done
        existing=$(find "$out_dir" -name "*_composited_audio.mp4" 2>/dev/null | wc -l)
        if [ "$existing" -gt 0 ]; then
            echo "  [SKIP] ${label}/${ds_name} ($existing composited videos)"
            round_jobs+=("${label}|${ds_name}|${out_dir}")
            job_idx=$((job_idx + 1))
            continue
        fi

        # Build -hp string
        hp="exp_path=${EXP_PATH},ckpt_name=step-${step}.pt,num_steps=${NUM_STEPS},guidance_scale=${cfg},output_dir=${out_dir},latentsync_inference=true,use_mouth_only_compositing=true,face_detection_cache_dir=${face_cache}"
        if [ "$noffo" = "true" ]; then
            hp="${hp},no_first_frame_overwrite=true"
        fi

        echo "  [GPU $gpu] ${label}/${ds_name} (cfg=${cfg}, noffo=${noffo})"
        mkdir -p "$out_dir"

        CUDA_VISIBLE_DEVICES=$gpu torchrun --standalone --nproc_per_node=1 \
            --master_port $((29500 + gpu)) \
            scripts/inference_v2v.py \
            --config configs/inference_v2v.yaml \
            --input_file "$input_file" \
            -hp "$hp" \
            > "${out_dir}.log" 2>&1 &
        pids+=($!)
        round_jobs+=("${label}|${ds_name}|${out_dir}")
        job_idx=$((job_idx + 1))
    done

    # Wait for this round
    for pid in "${pids[@]}"; do
        wait "$pid" || echo "WARNING: inference PID $pid failed"
    done

    # Post-process: move composited videos to clean directory
    for job_info in "${round_jobs[@]}"; do
        IFS='|' read -r label ds_name out_dir <<< "$job_info"
        comp_dir="${out_dir}_composited"
        if [ -d "$comp_dir" ] && [ "$(ls "$comp_dir"/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
            continue
        fi
        mkdir -p "$comp_dir"
        moved=0
        for f in "$out_dir"/*_composited_audio.mp4; do
            [ -f "$f" ] || continue
            stem=$(basename "$f" _composited_audio.mp4)
            mv "$f" "$comp_dir/${stem}.mp4"
            moved=$((moved + 1))
        done
        if [ $moved -gt 0 ]; then
            echo "  [POST] ${label}/${ds_name}: moved $moved → ${comp_dir##*/}"
        fi
    done

    echo "  Round done."
done
echo "  All inference complete."

# ─── Phase 2: Metrics ────────────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Metrics ==="
cd "$METRICS_REPO"

# Build metrics job list
METRIC_JOBS=()
for combo_entry in "${COMBOS[@]}"; do
    IFS='|' read -r label step cfg noffo <<< "$combo_entry"
    for ds_entry in "${DATASETS[@]}"; do
        IFS='|' read -r ds_name aligned_path orig_video_dir face_cache metrics_orig <<< "$ds_entry"

        comp_dir="${PROJECT_DIR}/${OUTPUT_BASE}/${label}/${ds_name}_composited"
        log_dir="${PROJECT_DIR}/${OUTPUT_BASE}/metrics/${label}/${ds_name}"

        # Determine real videos dir
        if [ "$ds_name" = "hallo3_mixed" ]; then
            real_dir="${PROJECT_DIR}/${MIXED_ORIG}"
        else
            real_dir="$metrics_orig"
        fi

        METRIC_JOBS+=("${label}|${ds_name}|${comp_dir}|${real_dir}|${log_dir}")
    done
done

# Run metrics in rounds
metric_idx=0
metric_total=${#METRIC_JOBS[@]}

while [ $metric_idx -lt $metric_total ]; do
    pids=()
    gpu_idx=0

    for gpu in "${GPUS[@]}"; do
        [ $metric_idx -ge $metric_total ] && break

        IFS='|' read -r label ds_name comp_dir real_dir log_dir <<< "${METRIC_JOBS[$metric_idx]}"

        # Skip if done
        if [ -f "$log_dir/metrics.log" ] && ! grep -q "failures" "$log_dir/metrics.log"; then
            echo "  [SKIP] metrics ${label}/${ds_name}"
            metric_idx=$((metric_idx + 1))
            continue
        fi

        # Skip if no composited videos
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
for combo_entry in "${COMBOS[@]}"; do
    IFS='|' read -r label step cfg noffo <<< "$combo_entry"
    for ds_name in hdtf hallo3 hallo3_mixed; do
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

# ─── Phase 3: Launch training ────────────────────────────────────────────────
echo ""
echo "=== Launching training from step-5500 ==="

# Pre-check: verify vae_latents_mask_all.pt exists
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
    --no_first_frame_overwrite
