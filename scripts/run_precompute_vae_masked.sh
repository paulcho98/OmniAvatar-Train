#!/bin/bash
# Precompute vae_latents_mask_all.pt for --mask_all_frames training.
# 4 GPUs in parallel, ~29K samples split into 4 shards.

VAE_PATH="pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
MASK_PATH="/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png"
DATA_LIST="/home/work/stableavatar_data/v2v_training_data/video_square_path.txt"

CUDA_VISIBLE_DEVICES=0 python scripts/precompute_vae_latents_masked.py \
    --vae_path "$VAE_PATH" --latentsync_mask_path "$MASK_PATH" \
    --data_list_path "$DATA_LIST" --shard_id 0 --num_shards 4 &

CUDA_VISIBLE_DEVICES=1 python scripts/precompute_vae_latents_masked.py \
    --vae_path "$VAE_PATH" --latentsync_mask_path "$MASK_PATH" \
    --data_list_path "$DATA_LIST" --shard_id 1 --num_shards 4 &

CUDA_VISIBLE_DEVICES=2 python scripts/precompute_vae_latents_masked.py \
    --vae_path "$VAE_PATH" --latentsync_mask_path "$MASK_PATH" \
    --data_list_path "$DATA_LIST" --shard_id 2 --num_shards 4 &

CUDA_VISIBLE_DEVICES=3 python scripts/precompute_vae_latents_masked.py \
    --vae_path "$VAE_PATH" --latentsync_mask_path "$MASK_PATH" \
    --data_list_path "$DATA_LIST" --shard_id 3 --num_shards 4 &

wait
echo "All shards complete."
