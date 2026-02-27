#!/bin/bash
# OmniAvatar V2V training with auxiliary losses (SyncNet + LPIPS).
# Requires: precomputed VAE latents + OmniAvatar audio embeddings.
# Based on StableAvatar's train_1B_square_v2v_pretrained_ref_auxloss.sh.
#
# VRAM: ~134 GB peak on H200 with --offload_frozen (required for aux losses).
# Without aux losses: ~61 GB peak, --offload_frozen not needed.

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate_2gpus.yaml \
    scripts/train_v2v.py \
    --dit_paths "pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors" \
    --text_encoder_path pretrained_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth \
    --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
    --wav2vec_path pretrained_models/wav2vec2-base-960h \
    --omniavatar_ckpt pretrained_models/OmniAvatar-14B/pytorch_model.pt \
    --data_list_path /home/work/stableavatar_data/v2v_training_data/video_square_path.txt \
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
    --output_path /home/work/output_omniavatar_v2v_auxloss \
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
    --validation_steps 1000 \
    --use_wandb \
    --wandb_entity "paulhcho" \
    --wandb_project "OmniAvatar-V2V" \
    --wandb_run_name "v2v_14B_auxloss" \
    --wandb_log_every 1 \
    --compute_sync_metrics 
