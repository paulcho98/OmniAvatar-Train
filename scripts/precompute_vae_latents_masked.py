"""Precompute VAE latents with frame 0 masked for --mask_all_frames training.

Reads existing vae_latents.pt (which has frame 0 unmasked in masked_latents),
loads the source video's frame 0, applies the LatentSync spatial mask, VAE-encodes
the masked frame 0, and saves as vae_latents_mask_all.pt.

The input_latents (unmasked full video) are copied as-is from the original file.

Usage:
    # 2 GPUs in parallel (same sharding as precompute_audio_omniavatar.py)
    CUDA_VISIBLE_DEVICES=0 python scripts/precompute_vae_latents_masked.py \
        --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
        --latentsync_mask_path /path/to/mask.png \
        --data_list_path /path/to/video_square_path.txt \
        --shard_id 0 --num_shards 2 &
    CUDA_VISIBLE_DEVICES=1 python scripts/precompute_vae_latents_masked.py \
        --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
        --latentsync_mask_path /path/to/mask.png \
        --data_list_path /path/to/video_square_path.txt \
        --shard_id 1 --num_shards 2 &
    wait
"""

import os
import sys
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.wan_video import WanVideoPipeline


def load_frame0_from_video(video_path, target_size=(512, 512)):
    """Load first frame from video, resize to target_size, return as PIL Image."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(frame)


def apply_mask_to_frame(frame_pil, mask_np):
    """Apply LatentSync mask to a single frame.

    Args:
        frame_pil: PIL Image (RGB, uint8)
        mask_np: [H, W] float32, 1=keep, 0=mask (LatentSync convention)
    Returns:
        masked_pil: PIL Image with mouth region zeroed
    """
    frame_np = np.array(frame_pil).astype(np.float32)
    h, w = frame_np.shape[:2]
    if mask_np.shape[0] != h or mask_np.shape[1] != w:
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask_resized > 0.5).astype(np.float32)
    else:
        mask_binary = (mask_np > 0.5).astype(np.float32)
    frame_np *= mask_binary[:, :, None]
    return Image.fromarray(frame_np.astype(np.uint8))


def encode_single_frame(vae, frame_pil, device, dtype):
    """VAE-encode a single frame. Returns [16, 1, H_lat, W_lat] tensor."""
    frame_np = np.array(frame_pil, dtype=np.float32) * (2.0 / 255.0) - 1.0
    frame_t = torch.from_numpy(frame_np).permute(2, 0, 1)  # [3, H, W]
    # Video tensor: [1, 3, 1, H, W]
    video_tensor = frame_t.unsqueeze(0).unsqueeze(2)
    latent = vae.encode(
        video_tensor.to(dtype=dtype, device=device),
        device=device, tiled=False,
    )
    return latent.squeeze(0).cpu()  # [16, 1, H_lat, W_lat]


def main():
    parser = argparse.ArgumentParser(description="Precompute mask-all-frames VAE latents")
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--latentsync_mask_path", type=str, required=True)
    parser.add_argument("--data_list_path", type=str, required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # Load VAE via ModelManager + Pipeline (same pattern as training code)
    print(f"[VAE] Loading from {args.vae_path}...")
    model_manager = ModelManager(device="cpu", infer=True)
    model_manager.load_model(args.vae_path, device="cpu", torch_dtype=dtype)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=dtype, device="cpu")
    vae = pipe.vae
    vae.to(device=device)
    print(f"[VAE] Loaded on {device}")

    # Load mask
    mask_img = Image.open(args.latentsync_mask_path).convert("L")
    mask_np = np.array(mask_img).astype(np.float32) / 255.0  # 1=keep, 0=mask

    # Load data list
    with open(args.data_list_path) as f:
        all_dirs = [line.strip() for line in f if line.strip()]

    # Shard
    shard_dirs = all_dirs[args.shard_id::args.num_shards]
    print(f"[Shard {args.shard_id}/{args.num_shards}] Processing {len(shard_dirs)}/{len(all_dirs)} directories")

    target_size = (args.target_size, args.target_size)
    skipped = 0
    processed = 0
    errors = 0

    for video_dir in tqdm(shard_dirs, desc=f"Shard {args.shard_id}"):
        output_path = os.path.join(video_dir, "vae_latents_mask_all.pt")

        # Skip if already exists (unless --force)
        if os.path.exists(output_path) and not args.force:
            skipped += 1
            continue

        # Load original vae_latents.pt
        orig_path = os.path.join(video_dir, "vae_latents.pt")
        if not os.path.exists(orig_path):
            print(f"[SKIP] No vae_latents.pt in {video_dir}")
            skipped += 1
            continue

        try:
            vae_data = torch.load(orig_path, map_location="cpu")
            input_latents = vae_data["input_latents"]    # [16, 21, 64, 64]
            masked_latents = vae_data["masked_latents"]   # [16, 21, 64, 64]

            # Load frame 0 from source video
            video_path = os.path.join(video_dir, "sub_clip.mp4")
            if not os.path.exists(video_path):
                video_path = os.path.join(video_dir, "video.mp4")
            if not os.path.exists(video_path):
                print(f"[SKIP] No video file in {video_dir}")
                skipped += 1
                continue

            frame0 = load_frame0_from_video(video_path, target_size)
            if frame0 is None:
                print(f"[ERROR] Failed to read frame 0 from {video_path}")
                errors += 1
                continue

            # Apply mask to frame 0
            masked_frame0 = apply_mask_to_frame(frame0, mask_np)

            # VAE-encode masked frame 0
            with torch.no_grad():
                masked_frame0_latent = encode_single_frame(vae, masked_frame0, device, dtype)
                # masked_frame0_latent: [16, 1, 64, 64]

            # Replace frame 0 in masked_latents
            new_masked_latents = masked_latents.clone()
            new_masked_latents[:, :1] = masked_frame0_latent  # Replace temporal frame 0

            # Save
            torch.save({
                "input_latents": input_latents,
                "masked_latents": new_masked_latents,
            }, output_path)
            processed += 1

        except Exception as e:
            print(f"[ERROR] {video_dir}: {e}")
            errors += 1

    print(f"\n[Done] Shard {args.shard_id}: processed={processed}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    main()
