#!/bin/bash
# Continue 14B V2V training: maskall + refseq + mouth_weight on 4 GPUs.
# Fresh optimizer, fresh wandb.
#
# === 14B Training Lineage ===
# Run 6: mouth_weight_4gpu (this script)
#   Source: mouth_weight_2gpu/step-1500.pt (run 5, cumulative 13000)
#   Output: /home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu/
#   Changes: 2GPU→4GPU, gradient_accumulation_steps 4→2
#
# Full lineage:
#   Run 1: auxloss                         → 5500 steps  (cumulative  5500)
#   Run 2: maskall_refseq                  → 1500 steps  (cumulative  7000)
#   Run 3: new_data_loss_weights           → 3000 steps  (cumulative 10000)
#   Run 4: new_data_loss_weights_mouth_wt  → 1500 steps  (cumulative 11500)
#   Run 5: mouth_weight_2gpu              → 1500 steps  (cumulative 13000)
#   Run 6: mouth_weight_4gpu (this)       → ...          (cumulative 13000+)

export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate_4gpus.yaml \
    scripts/train_v2v.py \
    --dit_paths "pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors" \
    --text_encoder_path pretrained_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth \
    --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
    --wav2vec_path pretrained_models/wav2vec2-base-960h \
    --omniavatar_ckpt /home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_2gpu/step-1500.pt \
    --data_list_path /home/work/stableavatar_data/v2v_training_data/video_square_path_combined.txt \
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
    --output_path /home/work/output_omniavatar_v2v_maskall_refseq_mouth_weight_4gpu \
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
    --validate_at_start \
    --use_wandb \
    --wandb_entity "paulhcho" \
    --wandb_project "OmniAvatar-V2V" \
    --wandb_run_name "v2v_14B_maskall_refseq_mouth_weight_4gpu_from13000" \
    --wandb_log_every 1 \
    --compute_sync_metrics \
    --offload_frozen \
    --mask_all_frames \
    --no_first_frame_overwrite \
    --use_ref_sequence \
    --wandb_api_key "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"
