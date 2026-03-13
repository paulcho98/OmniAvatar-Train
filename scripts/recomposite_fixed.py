#!/usr/bin/env python3
"""Re-composite 512x512 aligned face videos onto originals using fixed pipeline.

Fixes vs old compositing:
1. float32 in restore_img (already in affine_transform.py)
2. Float-space mouth blending (no uint8 quantization)
3. macro_block_size=None video saving (no resolution scaling)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/recomposite_fixed.py \
    --aligned_dir demo_out/v2v_eval_refseq_new/step-3000/hdtf_aligned \
    --originals_dir demo_out/v2v_eval_refseq_new/originals/hdtf \
    --face_cache_dir /path/to/face_cache \
    --output_dir demo_out/v2v_eval_refseq_new/step-3000/hdtf_composited_fixed
"""

import argparse
import os
import sys
import cv2
import subprocess
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF_v

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
    p.add_argument("--aligned_dir", required=True,
                   help="Dir with 512x512 face videos (.mp4)")
    p.add_argument("--input_suffix", default="",
                   help="Suffix to strip from input filenames to get the stem "
                        "(e.g., '_composited' if files are {stem}_composited.mp4)")
    p.add_argument("--originals_dir", required=True,
                   help="Dir with original full-res videos (.mp4)")
    p.add_argument("--face_cache_dir", required=True,
                   help="Dir with face detection caches (*_face_cache.pt)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--mask_path",
                   default="/home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fps", type=int, default=25)
    return p.parse_args()


def read_video_frames(path, max_frames=None):
    """Read frames as list of RGB numpy arrays."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        if max_frames and len(frames) >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_mouth_mask(mask_path, resolution):
    """Load mouth mask. 1.0=keep original (upper face), 0.0=use generated (mouth)."""
    mask = np.array(Image.open(mask_path).convert("L").resize(
        (resolution, resolution), Image.LANCZOS
    ))
    mask = torch.from_numpy(mask).float() / 255.0
    return mask.unsqueeze(0)  # [1, H, W]


def composite_float(aligned_frames, orig_frames, face_cache, restorer, mouth_mask):
    """Composite using float-space blending (no uint8 quantization).

    Matches inference_v2v.py:composite_with_latentsync_float() logic.
    """
    boxes = face_cache["boxes"]
    affine_matrices = face_cache["affine_matrices"]
    aligned_faces = face_cache.get("aligned_faces")  # [C, 512, 512] uint8 tensors

    composite_frames = []
    n = min(len(aligned_frames), len(orig_frames), len(boxes))

    for i in range(n):
        if boxes[i] is None:
            composite_frames.append(orig_frames[i])
            continue

        # Generated face as float [0, 1]
        face_float = torch.from_numpy(aligned_frames[i]).float() / 255.0  # [H, W, C]
        face_float = face_float.permute(2, 0, 1)  # [C, H, W]

        # Mouth-only compositing in float space
        if aligned_faces is not None and mouth_mask is not None:
            orig_aligned_float = aligned_faces[i].float() / 255.0  # [C, 512, 512]
            face_float = face_float * (1.0 - mouth_mask) + orig_aligned_float * mouth_mask

        # Resize to bounding box size
        x1, y1, x2, y2 = boxes[i]
        height, width = int(y2 - y1), int(x2 - x1)
        face_resized = TF_v.resize(
            face_float, size=[height, width],
            interpolation=TF_v.InterpolationMode.BICUBIC, antialias=True,
        )

        # Normalize to [-1, 1] for restore_img
        face_resized = face_resized * 2.0 - 1.0

        try:
            restored = restorer.restore_img(
                orig_frames[i], face_resized, affine_matrices[i]
            )
            composite_frames.append(restored)
        except Exception as e:
            print(f"  Frame {i} failed: {e}")
            composite_frames.append(orig_frames[i])

    return composite_frames


def save_video_fixed(frames, output_path, fps=25):
    """Save with imageio + macro_block_size=None (no resolution scaling)."""
    import imageio
    h, w = frames[0].shape[:2]
    writer = imageio.get_writer(
        str(output_path), fps=fps,
        macro_block_size=None,
        ffmpeg_params=["-crf", "13"],
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    restorer = AlignRestore(align_points=3, resolution=args.resolution, device=args.device)
    mouth_mask = load_mouth_mask(args.mask_path, args.resolution)
    print(f"Mouth mask: {mouth_mask.shape}")

    aligned_dir = Path(args.aligned_dir)
    originals_dir = Path(args.originals_dir)
    face_cache_dir = Path(args.face_cache_dir)

    # Find all aligned videos
    aligned_videos = sorted(aligned_dir.glob("*.mp4"))
    print(f"Found {len(aligned_videos)} aligned videos")

    for aligned_path in tqdm(aligned_videos, desc="Re-compositing"):
        stem = aligned_path.stem
        # Strip input suffix to get the base sample name
        if args.input_suffix and stem.endswith(args.input_suffix):
            stem = stem[:-len(args.input_suffix)]
        output_path = Path(args.output_dir) / f"{stem}.mp4"

        if output_path.exists():
            print(f"  [SKIP] {stem}")
            continue

        # Find corresponding original
        orig_path = originals_dir / f"{stem}.mp4"
        if not orig_path.exists():
            print(f"  [SKIP] {stem} (no original)")
            continue

        # Load face cache
        cache_path = face_cache_dir / f"{stem}_face_cache.pt"
        if not cache_path.exists():
            print(f"  [SKIP] {stem} (no face cache)")
            continue

        cache = torch.load(cache_path, weights_only=False)

        # Read frames
        aligned_frames = read_video_frames(aligned_path)
        orig_frames = read_video_frames(orig_path)

        # Composite with float precision
        composited = composite_float(
            aligned_frames, orig_frames, cache, restorer, mouth_mask
        )

        # Save with fixed encoding (no macro_block_size scaling)
        save_video_fixed(composited, output_path, fps=args.fps)

        # Verify resolution matches original
        out_h, out_w = composited[0].shape[:2]
        orig_h, orig_w = orig_frames[0].shape[:2]
        if out_h != orig_h or out_w != orig_w:
            print(f"  WARNING: {stem} output={out_w}x{out_h} vs original={orig_w}x{orig_h}")

    print(f"\nDone. {len(list(Path(args.output_dir).glob('*.mp4')))} videos in {args.output_dir}")


if __name__ == "__main__":
    main()
