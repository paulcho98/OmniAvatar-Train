#!/usr/bin/env python3
"""
Standalone compositing: paste generated 512x512 face videos back onto
original-resolution source videos using inverse affine transforms.

No diffusion model needed — just InsightFace + kornia.
Face detection is cached per-original-video and reused across checkpoints.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/composite_v2v.py \
    --generated_dir demo_out/v2v_eval/step-5000/hdtf \
    --original_videos_dir /home/work/.local/HDTF/HDTF_original_testset_81frames/videos_cfr \
    --output_dir demo_out/v2v_eval/step-5000/hdtf_composited \
    --face_cache_dir demo_out/v2v_eval/face_cache/hdtf \
    --existing_face_cache_dir /home/work/.local/StableAvatar/validation_hdtf/shared_hdtf_data/face_cache \
    --num_frames 81 --fps 25
"""

import argparse, os, sys, cv2, subprocess, torch, numpy as np
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms.functional as TF_v

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import AlignRestore directly to avoid insightface dependency in __init__.py
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "affine_transform",
    os.path.join(os.path.dirname(__file__), "..", "OmniAvatar", "utils", "latentsync", "affine_transform.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AlignRestore = _mod.AlignRestore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generated_dir", required=True,
                   help="Dir with generated *_audio.mp4 (or *_cfr25_audio.mp4)")
    p.add_argument("--original_videos_dir", required=True,
                   help="Dir with original full-res .mp4 videos")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--face_cache_dir", required=True,
                   help="Cache dir for new face detection (shared across checkpoints)")
    p.add_argument("--existing_face_cache_dir", default=None,
                   help="Pre-existing face cache dir to load from (avoids recomputation)")
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--resolution", type=int, default=512,
                   help="Resolution of generated (aligned) face videos")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def read_video_frames(path, num_frames):
    """Read up to num_frames from video, pad last frame if short."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames


def build_name_mapping(generated_dir, original_videos_dir):
    """Map generated video paths to their corresponding original video paths.

    Generated names: {sample}_cfr25_audio.mp4 or {sample}_audio.mp4
    Original names: {sample}_cfr25.mp4 or {sample}.mp4
    """
    gen_files = sorted(Path(generated_dir).glob("*_audio.mp4"))
    orig_dir = Path(original_videos_dir)
    orig_files = {p.stem: p for p in orig_dir.glob("*.mp4")}

    mapping = []
    for gen_path in gen_files:
        # Strip _audio suffix to get the sample name
        stem = gen_path.stem
        if stem.endswith("_audio"):
            stem = stem[:-6]  # remove "_audio"

        # Try exact match first, then with/without _cfr25
        orig_path = None
        for candidate in [stem, stem + "_cfr25", stem.replace("_cfr25", "")]:
            if candidate in orig_files:
                orig_path = orig_files[candidate]
                break

        if orig_path is None:
            print(f"WARNING: No original found for {gen_path.name}, skipping")
            continue

        mapping.append((gen_path, orig_path, stem))

    return mapping


def load_face_cache(face_cache_dir, stem, resolution, existing_face_cache_dir=None):
    """Load face detection cache. Checks existing cache dir first, then local cache.

    Returns None if no cache found (fresh detection not supported in standalone mode —
    install insightface and use inference_v2v.py with latentsync_inference=true instead).
    """
    cache_filename = f"{stem}_face_cache.pt"

    # 1. Check pre-existing cache (from StableAvatar evaluation)
    if existing_face_cache_dir:
        existing_path = Path(existing_face_cache_dir) / cache_filename
        if existing_path.exists():
            cache = torch.load(existing_path, weights_only=False)
            if cache.get("resolution") == resolution:
                print(f"  Loaded existing cache: {existing_path}")
                return cache

    # 2. Check local cache (computed in prior runs)
    cache_path = Path(face_cache_dir) / cache_filename
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=False)
        if cache.get("resolution") == resolution:
            print(f"  Loaded local cache: {cache_path.name}")
            return cache

    print(f"  ERROR: No face cache found for {stem}")
    return None


def composite_single_video(gen_path, original_path, face_cache, restorer, num_frames):
    """Composite generated face video back onto original frames.

    Replicates logic from inference_v2v.py:composite_with_latentsync (lines 185-252).
    """
    gen_frames = read_video_frames(str(gen_path), num_frames)
    orig_frames = read_video_frames(str(original_path), num_frames)

    boxes = face_cache["boxes"]
    affine_matrices = face_cache["affine_matrices"]
    composite_frames = []

    for i in range(min(len(gen_frames), len(orig_frames))):
        if boxes[i] is None:
            composite_frames.append(orig_frames[i])
            continue

        # HWC uint8 -> CHW tensor
        face_tensor = torch.from_numpy(gen_frames[i]).permute(2, 0, 1)

        # Resize generated face to match bounding box size
        x1, y1, x2, y2 = boxes[i]
        height, width = int(y2 - y1), int(x2 - x1)
        face_resized = TF_v.resize(
            face_tensor, size=[height, width],
            interpolation=TF_v.InterpolationMode.BICUBIC, antialias=True,
        )

        # Normalize to [-1, 1] for restore_img
        face_resized = face_resized.float() / 255.0 * 2.0 - 1.0

        try:
            restored = restorer.restore_img(
                orig_frames[i], face_resized, affine_matrices[i]
            )
            composite_frames.append(restored)
        except Exception as e:
            print(f"  Restoration failed frame {i}: {e}")
            composite_frames.append(orig_frames[i])

    return composite_frames


def save_video_with_audio(frames, audio_source_path, output_path, fps=25):
    """Save frames as video, extract audio from source, mux together."""
    h, w = frames[0].shape[:2]
    tmp_video = str(output_path) + ".tmp.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    # Mux with audio from the generated _audio.mp4
    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_video, "-i", str(audio_source_path),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
        str(output_path),
    ], capture_output=True)

    os.remove(tmp_video)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.face_cache_dir, exist_ok=True)

    # Initialize AlignRestore (only needs kornia, no InsightFace)
    restorer = AlignRestore(align_points=3, resolution=args.resolution, device=args.device)

    # Build mapping: generated -> original
    mapping = build_name_mapping(args.generated_dir, args.original_videos_dir)
    print(f"Found {len(mapping)} videos to composite")

    for gen_path, orig_path, stem in tqdm(mapping, desc="Compositing"):
        output_path = Path(args.output_dir) / f"{stem}.mp4"
        if output_path.exists():
            print(f"  Skipping {stem} (exists)")
            continue

        print(f"  {stem}: {gen_path.name} -> {orig_path.name}")

        # Load face cache (pre-existing StableAvatar caches preferred)
        cache = load_face_cache(
            args.face_cache_dir, stem, args.resolution,
            existing_face_cache_dir=args.existing_face_cache_dir,
        )
        if cache is None:
            print(f"  SKIP: no face cache for {stem}")
            continue

        # Composite
        frames = composite_single_video(
            gen_path, orig_path, cache, restorer, args.num_frames
        )

        # Save with audio
        save_video_with_audio(frames, gen_path, output_path, args.fps)
        print(f"  Saved: {output_path}")

    print(f"Done. {len(list(Path(args.output_dir).glob('*.mp4')))} composited videos in {args.output_dir}")


if __name__ == "__main__":
    main()
