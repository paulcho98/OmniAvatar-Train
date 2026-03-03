"""Precompute VAE latents with frame 0 masked for --mask_all_frames training.

Reads existing vae_latents.pt (which has frame 0 unmasked in masked_latents),
loads the source video's frame 0, applies the LatentSync spatial mask, VAE-encodes
the masked frame 0, and saves as vae_latents_mask_all.pt.

The input_latents (unmasked full video) are copied as-is from the original file.

Uses threaded I/O: a pool pre-loads the next batch while the GPU encodes the
current one, and writes outputs in the background.

Usage:
    # 4 GPUs in parallel
    bash scripts/run_precompute_vae_masked.sh
"""

import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.wan_video import WanVideoPipeline


def load_frame0_from_video(video_path, target_size=(512, 512)):
    """Load first frame from video, resize to target_size."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
    return frame  # numpy uint8 [H, W, 3]


def load_one_sample(video_dir, target_size, mask_binary, force):
    """Load and prepare one sample (CPU-only, thread-safe).

    Returns (video_dir, frame_tensor, input_latents, masked_latents) or None.
    """
    output_path = os.path.join(video_dir, "vae_latents_mask_all.pt")
    if os.path.exists(output_path) and not force:
        return "skip"

    orig_path = os.path.join(video_dir, "vae_latents.pt")
    if not os.path.exists(orig_path):
        return "skip"

    vae_data = torch.load(orig_path, map_location="cpu")
    input_latents = vae_data["input_latents"]    # [16, 21, 64, 64]
    masked_latents = vae_data["masked_latents"]   # [16, 21, 64, 64]

    video_path = os.path.join(video_dir, "sub_clip.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join(video_dir, "video.mp4")
    if not os.path.exists(video_path):
        return "skip"

    frame0 = load_frame0_from_video(video_path, target_size)
    if frame0 is None:
        return "error"

    # Normalize to [-1,1] THEN mask → masked region = 0.0 (matches precompute convention)
    frame_np = frame0.astype(np.float32) * (2.0 / 255.0) - 1.0
    frame_np *= mask_binary[:, :, None]
    frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(1)  # [3, 1, H, W]

    return (video_dir, frame_tensor, input_latents, masked_latents)


def save_one_sample(video_dir, input_latents, new_masked_latents):
    """Save result (CPU-only, thread-safe)."""
    output_path = os.path.join(video_dir, "vae_latents_mask_all.pt")
    torch.save({
        "input_latents": input_latents,
        "masked_latents": new_masked_latents,
    }, output_path)


def main():
    parser = argparse.ArgumentParser(description="Precompute mask-all-frames VAE latents")
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--latentsync_mask_path", type=str, required=True)
    parser.add_argument("--data_list_path", type=str, required=True)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of frames to VAE-encode at once")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="I/O threads for loading and saving")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # Load VAE
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

    shard_dirs = all_dirs[args.shard_id::args.num_shards]
    print(f"[Shard {args.shard_id}/{args.num_shards}] {len(shard_dirs)}/{len(all_dirs)} dirs, "
          f"batch={args.batch_size}, workers={args.num_workers}")

    skipped = 0
    processed = 0
    errors = 0

    load_pool = ThreadPoolExecutor(max_workers=args.num_workers)
    save_pool = ThreadPoolExecutor(max_workers=args.num_workers)
    save_futures = []

    # Submit all loads upfront as futures, process in batch-sized chunks
    pbar = tqdm(total=len(shard_dirs), desc=f"Shard {args.shard_id}")

    batch_tensors = []
    batch_meta = []  # (video_dir, input_latents, masked_latents)

    def flush_batch():
        nonlocal processed, errors
        if not batch_tensors:
            return
        stacked = torch.stack(batch_tensors, dim=0)  # [B, 3, 1, H, W]
        try:
            with torch.no_grad():
                latents = vae.encode(
                    stacked.to(dtype=dtype, device=device),
                    device=device, tiled=False,
                )  # [B, 16, 1, H_lat, W_lat]
            latents_cpu = latents.cpu()
            for i, (video_dir, input_lat, masked_lat) in enumerate(batch_meta):
                new_masked = masked_lat.clone()
                new_masked[:, :1] = latents_cpu[i]
                fut = save_pool.submit(save_one_sample, video_dir, input_lat, new_masked)
                save_futures.append(fut)
                processed += 1
        except Exception as e:
            print(f"[ERROR] Batch encode failed: {e}")
            errors += len(batch_meta)
        batch_tensors.clear()
        batch_meta.clear()

    # Use thread pool to pre-load data
    # Process in chunks: submit batch_size loads, gather results, encode, repeat
    idx = 0
    while idx < len(shard_dirs):
        # Submit a chunk of loads
        chunk_end = min(idx + args.batch_size, len(shard_dirs))
        chunk_dirs = shard_dirs[idx:chunk_end]
        futures = {
            load_pool.submit(load_one_sample, d, target_size, mask_binary, args.force): d
            for d in chunk_dirs
        }

        for fut in as_completed(futures):
            pbar.update(1)
            try:
                result = fut.result()
            except Exception as e:
                print(f"[ERROR] Load failed: {e}")
                errors += 1
                continue

            if result == "skip":
                skipped += 1
                continue
            if result == "error":
                errors += 1
                continue

            video_dir, frame_tensor, input_lat, masked_lat = result
            batch_tensors.append(frame_tensor)
            batch_meta.append((video_dir, input_lat, masked_lat))

        # Encode the accumulated batch
        flush_batch()
        idx = chunk_end

    flush_batch()

    # Wait for all saves to complete
    for fut in save_futures:
        fut.result()

    pbar.close()
    load_pool.shutdown()
    save_pool.shutdown()

    print(f"\n[Done] Shard {args.shard_id}: processed={processed}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    main()
