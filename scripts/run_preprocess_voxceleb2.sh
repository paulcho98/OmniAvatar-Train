#!/bin/bash
# Preprocess VoxCeleb2 videos for V2V training.
# Existing 29K entries are skipped (all files present).
# Processes ~7,453 new VoxCeleb2 videos on 4 GPUs.
#
# Usage: bash scripts/run_preprocess_voxceleb2.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/work/.local/miniconda3/envs/omniavatar/bin/python \
    scripts/preprocess_v2v_integrated.py \
    --csv_path /home/work/stableavatar_data/v2v_training_data/metadata_combined_with_voxceleb2.csv \
    --video_dir /home/work/data/VoxCeleb2_processed_all/high_visual_quality \
    --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
    --wav2vec_path pretrained_models/wav2vec2-base-960h \
    --latentsync_mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
    --output_dir /home/work/stableavatar_data/v2v_training_data/ \
    --num_frames 81 --height 512 --width 512 \
    --batch_size 4 \
    --num_cpu_workers 16
