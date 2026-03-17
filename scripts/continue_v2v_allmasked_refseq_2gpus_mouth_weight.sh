#!/bin/bash
# Continue OmniAvatar V2V training (fresh optimizer, fresh wandb).
# 2-GPU variant with mouth_weight=2.0 and updated loss weights.
# Uses trainable_weights.pt from the latest checkpoint dir (auto-detected).
#
# No Accelerate --resume_from_checkpoint (4-GPU checkpoint incompatible with 2-GPU).
#
# === 14B Training Lineage ===
# Run 5: mouth_weight_2gpu (this script)
#   Source: new_data_loss_weights_mouth_weights/checkpoint-1500/trainable_weights.pt (run 4, cumulative 11500)
#   Output: /home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_2gpu/ (1500 steps → cumulative 13000)
#   Changes: 4GPU→2GPU, gradient_accumulation_steps 2→4

OUTPUT_DIR=/home/work/output_omniavatar_v2v_maskall_refseq_new_data_loss_weights_mouth_weights

# Auto-detect latest checkpoint
LATEST_CKPT=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in $OUTPUT_DIR"
    exit 1
fi
CKPT_FILE="$LATEST_CKPT/trainable_weights.pt"
if [ ! -f "$CKPT_FILE" ]; then
    echo "ERROR: $CKPT_FILE not found"
    exit 1
fi
STEP=$(basename "$LATEST_CKPT" | grep -oP '\d+')
echo "Resuming from $CKPT_FILE (step $STEP)"

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate_2gpus.yaml \
    scripts/train_v2v.py \
    --dit_paths "pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors" \
    --text_encoder_path pretrained_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth \
    --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
    --wav2vec_path pretrained_models/wav2vec2-base-960h \
    --omniavatar_ckpt "$CKPT_FILE" \
    --data_list_path /home/work/stableavatar_data/v2v_training_data/video_square_path_combined.txt \
    --latentsync_mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
    --use_precomputed_vae \
    --use_precomputed_audio \
    --use_precomputed_text_emb \
    --num_frames 81 --height 512 --width 512 \
    --num_epochs 1000 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 1 \
    --weight_decay 0.01 \
    --seed 42 \
    --use_gradient_checkpointing \
    --text_drop_prob 0.1 \
    --audio_drop_prob 0.1 \
    --output_path /home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_2gpu \
    --save_steps 500 \
    --use_sync_loss \
    --use_lpips_loss \
    --use_trepa_loss \
    --use_vae_gradient_checkpointing \
    --aux_recon_weight 1.0 \
    --aux_sync_weight 0.05 \
    --aux_lpips_weight 0.15 \
    --aux_trepa_weight 10.0 \
    --aux_num_frames 21 \
    --mouth_weight 2.0 \
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
    --wandb_run_name "v2v_14B_maskall_refseq_mouth_weight_2gpu_from${STEP}" \
    --wandb_log_every 1 \
    --compute_sync_metrics \
    --offload_frozen \
    --mask_all_frames \
    --no_first_frame_overwrite \
    --use_ref_sequence \
    --wandb_api_key "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"
