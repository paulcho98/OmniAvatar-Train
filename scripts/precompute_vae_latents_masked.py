"""Precompute VAE latents with frame 0 masked for --mask_all_frames training.

Reads existing vae_latents.pt (which has frame 0 unmasked in masked_latents),
loads the source video's frame 0, applies the LatentSync spatial mask, VAE-encodes
the masked frame 0, and saves as vae_latents_mask_all.pt.

The input_latents (unmasked full video) are copied as-is from the original file.

Usage:
    # 4 GPUs in parallel
    bash scripts/run_precompute_vae_masked.sh
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


def frame_to_masked_tensor(frame_pil, mask_binary):
    """Normalize to [-1,1] THEN mask. Masked region = 0.0 (mid-gray), not -1.0 (black).

    Must match StableAvatar's precompute convention (normalize first, mask second).
    Returns [3, 1, H, W] float32 tensor.
    """
    frame_np = np.array(frame_pil).astype(np.float32) * (2.0 / 255.0) - 1.0  # [-1, 1]
    frame_np *= mask_binary[:, :, None]  # masked region → 0.0
    t = torch.from_numpy(frame_np).permute(2, 0, 1)  # [3, H, W]
    return t.unsqueeze(1)  # [3, 1, H, W]


def main():
    parser = argparse.ArgumentParser(description="Precompute mask-all-frames VAE latents")
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--latentsync_mask_path", type=str, required=True)
    parser.add_argument("--data_list_path", type=str, required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of frames to VAE-encode at once")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # Load VAE via ModelManager + Pipeline
    print(f"[VAE] Loading from {args.vae_path}...")
    model_manager = ModelManager(device="cpu", infer=True)
    model_manager.load_model(args.vae_path, device="cpu", torch_dtype=dtype)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=dtype, device="cpu")
    vae = pipe.vae
    vae.to(device=device)
    print(f"[VAE] Loaded on {device}")

    # Load and pre-binarize mask
    mask_img = Image.open(args.latentsync_mask_path).convert("L")
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    target_size = (args.target_size, args.target_size)
    if mask_np.shape[0] != target_size[0] or mask_np.shape[1] != target_size[1]:
        mask_np = cv2.resize(mask_np, target_size, interpolation=cv2.INTER_LINEAR)
    mask_binary = (mask_np > 0.5).astype(np.float32)

    # Load data list
    with open(args.data_list_path) as f:
        all_dirs = [line.strip() for line in f if line.strip()]

    # Shard
    shard_dirs = all_dirs[args.shard_id::args.num_shards]
    print(f"[Shard {args.shard_id}/{args.num_shards}] Processing {len(shard_dirs)}/{len(all_dirs)} dirs, batch_size={args.batch_size}")

    skipped = 0
    processed = 0
    errors = 0

    # Collect batches: prepare CPU data, then batch-encode on GPU
    batch_tensors = []   # masked frame tensors ready for VAE
    batch_meta = []      # (video_dir, input_latents, masked_latents) per item

    def flush_batch():
        nonlocal processed, errors
        if not batch_tensors:
            return
        # Stack into [B, 3, 1, H, W] and encode
        stacked = torch.stack(batch_tensors, dim=0)  # [B, 3, 1, H, W]
        try:
            with torch.no_grad():
                # vae.encode expects [B, C, T, H, W], iterates over B
                latents = vae.encode(
                    stacked.to(dtype=dtype, device=device),
                    device=device, tiled=False,
                )  # [B, 16, 1, H_lat, W_lat]
            latents = latents.cpu()
            for i, (video_dir, input_lat, masked_lat) in enumerate(batch_meta):
                new_masked = masked_lat.clone()
                new_masked[:, :1] = latents[i]  # Replace temporal frame 0
                output_path = os.path.join(video_dir, "vae_latents_mask_all.pt")
                torch.save({
                    "input_latents": input_lat,
                    "masked_latents": new_masked,
                }, output_path)
                processed += 1
        except Exception as e:
            print(f"[ERROR] Batch encode failed: {e}")
            errors += len(batch_meta)
        batch_tensors.clear()
        batch_meta.clear()

    for video_dir in tqdm(shard_dirs, desc=f"Shard {args.shard_id}"):
        output_path = os.path.join(video_dir, "vae_latents_mask_all.pt")

        if os.path.exists(output_path) and not args.force:
            skipped += 1
            continue

        orig_path = os.path.join(video_dir, "vae_latents.pt")
        if not os.path.exists(orig_path):
            skipped += 1
            continue

        try:
            vae_data = torch.load(orig_path, map_location="cpu")
            input_latents = vae_data["input_latents"]    # [16, 21, 64, 64]
            masked_latents = vae_data["masked_latents"]   # [16, 21, 64, 64]

            video_path = os.path.join(video_dir, "sub_clip.mp4")
            if not os.path.exists(video_path):
                video_path = os.path.join(video_dir, "video.mp4")
            if not os.path.exists(video_path):
                skipped += 1
                continue

            frame0 = load_frame0_from_video(video_path, target_size)
            if frame0 is None:
                errors += 1
                continue

            frame_tensor = frame_to_masked_tensor(frame0, mask_binary)  # [3, 1, H, W]

            batch_tensors.append(frame_tensor)
            batch_meta.append((video_dir, input_latents, masked_latents))

            if len(batch_tensors) >= args.batch_size:
                flush_batch()

        except Exception as e:
            print(f"[ERROR] {video_dir}: {e}")
            errors += 1

    # Flush remaining
    flush_batch()

    print(f"\n[Done] Shard {args.shard_id}: processed={processed}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    main()
