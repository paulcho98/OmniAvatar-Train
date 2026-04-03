#!/usr/bin/env python3
"""Evaluation with GT-aligned face crops.

Uses face detection from GT video only, applies same bounding box to both
GT and generated videos. Eliminates MediaPipe crop misalignment artifacts.

Computes: SSIM, FID, FVD (all using GT bboxes for both videos).

Usage:
  python scripts/eval_aligned_crops.py \
    --real_videos_dir path/to/originals \
    --fake_videos_dir path/to/composited \
    --output_dir path/to/metrics_aligned_crops
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_f


class GTFaceCropper:
    """Detects faces in GT frames, uses same bbox for both GT and generated."""

    def __init__(self, resolution=(224, 224), min_confidence=0.5, fallback_confidence=0.2):
        self.resolution = resolution
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_confidence
        )
        self.fallback = None
        if fallback_confidence < min_confidence:
            self.fallback = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=fallback_confidence
            )

    def detect(self, img_bgr):
        """Detect face bbox in a single frame. Returns (xmin, ymin, xmax, ymax) or None."""
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        if not results.detections and self.fallback:
            results = self.fallback.process(rgb)
        if not results.detections:
            return None
        bb = results.detections[0].location_data.relative_bounding_box
        xmin = max(0, int(bb.xmin * w))
        ymin = max(0, int(bb.ymin * h))
        xmax = min(w, xmin + int(bb.width * w))
        ymax = min(h, ymin + int(bb.height * h))
        return (xmin, ymin, xmax, ymax)

    def crop(self, img_bgr, bbox):
        """Crop and resize using a given bbox."""
        xmin, ymin, xmax, ymax = bbox
        face = img_bgr[ymin:ymax, xmin:xmax]
        return cv2.resize(face, (self.resolution[1], self.resolution[0]),
                          interpolation=cv2.INTER_AREA)


def read_video_frames(path, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        if max_frames and len(frames) >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _strip_cfr25(stem):
    return stem[:-len("_cfr25")] if stem.endswith("_cfr25") else stem


def _delimited_match(stem, key):
    """Check if key appears in stem as a delimited substring."""
    import re
    pattern = r"(^|[_.-]){}($|[_.-])".format(re.escape(key))
    return re.search(pattern, stem) is not None


def match_videos(real_dir, fake_dir):
    """Match real and fake videos by name using delimited key matching.

    Uses the same matching logic as the standard eval pipeline's video_key.py:
    normalize GT stems (strip _cfr25), then find each GT key inside fake stems
    via delimited matching (handles _synced, _wav2lip, _retalking, etc.).
    """
    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)
    real_videos = {p.stem: p for p in sorted(real_dir.glob("*.mp4"))}
    fake_videos = {p.stem: p for p in sorted(fake_dir.glob("*.mp4"))}

    # Build normalized GT keys
    known_keys = {_strip_cfr25(stem): stem for stem in real_videos}

    pairs = []
    # For each GT video, find a matching fake video
    for gt_key, orig_stem in known_keys.items():
        real_path = real_videos[orig_stem]
        # Try exact matches first
        for candidate in [orig_stem, gt_key, gt_key + "_cfr25"]:
            if candidate in fake_videos:
                pairs.append((real_path, fake_videos[candidate], gt_key))
                break
        else:
            # Delimited search: find gt_key inside fake stems
            for fake_stem, fake_path in fake_videos.items():
                if _delimited_match(fake_stem, gt_key):
                    pairs.append((real_path, fake_path, gt_key))
                    break

    return pairs


# ─── SSIM ────────────────────────────────────────────────────────────────────

def compute_ssim_aligned(pairs, cropper, max_frames=None):
    """Compute SSIM using GT bboxes for both videos."""
    all_scores = []
    per_video = {}

    for real_path, fake_path, stem in tqdm(pairs, desc="SSIM"):
        real_frames = read_video_frames(real_path, max_frames)
        fake_frames = read_video_frames(fake_path, max_frames)
        n = min(len(real_frames), len(fake_frames))

        scores = []
        skipped = 0
        for i in range(n):
            bbox = cropper.detect(real_frames[i])  # detect in GT only
            if bbox is None:
                skipped += 1
                continue

            # Resize fake to match real if needed
            fake_frame = fake_frames[i]
            if fake_frame.shape[:2] != real_frames[i].shape[:2]:
                fake_frame = cv2.resize(fake_frame,
                                        (real_frames[i].shape[1], real_frames[i].shape[0]),
                                        interpolation=cv2.INTER_AREA)

            real_crop = cropper.crop(real_frames[i], bbox)
            fake_crop = cropper.crop(fake_frame, bbox)  # same bbox!

            real_gray = cv2.cvtColor(real_crop, cv2.COLOR_BGR2GRAY)
            fake_gray = cv2.cvtColor(fake_crop, cv2.COLOR_BGR2GRAY)

            s = ssim_f(real_gray, fake_gray)
            scores.append(s)

        if scores:
            mean_s = np.mean(scores)
            all_scores.extend(scores)
            per_video[stem] = mean_s

    return np.mean(all_scores) if all_scores else 0, per_video


# ─── FID ─────────────────────────────────────────────────────────────────────

def extract_crops_to_dir(pairs, cropper, real_out, fake_out, max_frames=None):
    """Extract GT-bbox-aligned crops to directories for FID computation."""
    os.makedirs(real_out, exist_ok=True)
    os.makedirs(fake_out, exist_ok=True)

    total_real = 0
    total_fake = 0

    for real_path, fake_path, stem in tqdm(pairs, desc="Extracting crops"):
        real_frames = read_video_frames(real_path, max_frames)
        fake_frames = read_video_frames(fake_path, max_frames)
        n = min(len(real_frames), len(fake_frames))

        for i in range(n):
            bbox = cropper.detect(real_frames[i])
            if bbox is None:
                continue

            fake_frame = fake_frames[i]
            if fake_frame.shape[:2] != real_frames[i].shape[:2]:
                fake_frame = cv2.resize(fake_frame,
                                        (real_frames[i].shape[1], real_frames[i].shape[0]),
                                        interpolation=cv2.INTER_AREA)

            real_crop = cropper.crop(real_frames[i], bbox)
            fake_crop = cropper.crop(fake_frame, bbox)

            cv2.imwrite(os.path.join(real_out, f"{stem}_{i:04d}.png"), real_crop)
            cv2.imwrite(os.path.join(fake_out, f"{stem}_{i:04d}.png"), fake_crop)
            total_real += 1
            total_fake += 1

    return total_real, total_fake


def compute_fid(real_crops_dir, fake_crops_dir, device="cuda:0"):
    """Compute FID between two directories of images."""
    from pytorch_fid import fid_score
    fid = fid_score.calculate_fid_given_paths(
        [real_crops_dir, fake_crops_dir],
        batch_size=64, device=device, dims=2048
    )
    return fid


# ─── FVD ─────────────────────────────────────────────────────────────────────

def compute_fvd_aligned(pairs, cropper, i3d_path, device="cuda:0",
                         frame_start=20, frame_end=36):
    """Compute FVD using GT bboxes, I3D features on frames [start:end]."""
    num_frames = frame_end - frame_start

    # Load I3D model
    with open(i3d_path, "rb") as f:
        i3d = torch.jit.load(f).eval().to(device)
    i3d_kwargs = dict(rescale=False, resize=False, return_features=True)

    real_features = []
    fake_features = []

    for real_path, fake_path, stem in tqdm(pairs, desc="FVD"):
        real_frames = read_video_frames(real_path)
        fake_frames = read_video_frames(fake_path)

        if len(real_frames) < frame_end or len(fake_frames) < frame_end:
            continue

        real_clips = []
        fake_clips = []

        for i in range(frame_start, frame_end):
            bbox = cropper.detect(real_frames[i])
            if bbox is None:
                break

            fake_frame = fake_frames[i]
            if fake_frame.shape[:2] != real_frames[i].shape[:2]:
                fake_frame = cv2.resize(fake_frame,
                                        (real_frames[i].shape[1], real_frames[i].shape[0]),
                                        interpolation=cv2.INTER_AREA)

            real_crop = cropper.crop(real_frames[i], bbox)
            fake_crop = cropper.crop(fake_frame, bbox)

            # BGR → RGB, normalize to [0, 1]
            real_clips.append(cv2.cvtColor(real_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            fake_clips.append(cv2.cvtColor(fake_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)

        if len(real_clips) < num_frames:
            continue

        # [T, H, W, C] → [1, C, T, H, W] (I3D expects channel-first)
        real_np = np.stack(real_clips)  # [T, H, W, C]
        fake_np = np.stack(fake_clips)
        real_tensor = torch.from_numpy(real_np).permute(3, 0, 1, 2).unsqueeze(0).to(device)  # [1, C, T, H, W]
        fake_tensor = torch.from_numpy(fake_np).permute(3, 0, 1, 2).unsqueeze(0).to(device)

        with torch.no_grad():
            real_feat = i3d(real_tensor, **i3d_kwargs).cpu().numpy()
            fake_feat = i3d(fake_tensor, **i3d_kwargs).cpu().numpy()

        real_features.append(real_feat[0])
        fake_features.append(fake_feat[0])

    if not real_features:
        return None

    real_features = np.stack(real_features)
    fake_features = np.stack(fake_features)

    # FVD = Fréchet distance between feature distributions
    from scipy.linalg import sqrtm
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    covmean, _ = sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fvd = float(np.square(diff).sum() + np.trace(sigma_real + sigma_fake - 2 * covmean))
    return fvd


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Eval with GT-aligned crops')
    parser.add_argument('--real_videos_dir', required=True)
    parser.add_argument('--fake_videos_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--metrics', nargs='+', default=['ssim', 'fid', 'fvd'],
                        choices=['ssim', 'fid', 'fvd'])
    parser.add_argument('--crop_resolution', type=int, nargs=2, default=[224, 224])
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--i3d_path', default='/home/work/.local/latentsync-metrics/checkpoints/auxiliary/i3d_torchscript.pt')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cropper = GTFaceCropper(resolution=tuple(args.crop_resolution))
    pairs = match_videos(args.real_videos_dir, args.fake_videos_dir)
    print(f"Matched {len(pairs)} video pairs")

    results = {}

    # SSIM
    if 'ssim' in args.metrics:
        print("\n=== SSIM (GT-aligned crops) ===")
        mean_ssim, per_video = compute_ssim_aligned(pairs, cropper, args.max_frames)
        results['SSIM'] = mean_ssim
        print(f"Mean SSIM: {mean_ssim:.6f}")

        with open(os.path.join(args.output_dir, 'ssim_per_video.log'), 'w') as f:
            for stem, s in sorted(per_video.items()):
                f.write(f"{stem}: SSIM={s:.6f}\n")
            f.write(f"mean_ssim: {mean_ssim:.6f}\n")

    # FID
    if 'fid' in args.metrics:
        print("\n=== FID (GT-aligned crops) ===")
        real_crops = os.path.join(args.output_dir, 'crops_real')
        fake_crops = os.path.join(args.output_dir, 'crops_fake')
        n_real, n_fake = extract_crops_to_dir(pairs, cropper, real_crops, fake_crops, args.max_frames)
        print(f"Extracted {n_real} real, {n_fake} fake crops")
        fid = compute_fid(real_crops, fake_crops, args.device)
        results['FID'] = fid
        print(f"FID: {fid:.4f}")

    # FVD
    if 'fvd' in args.metrics:
        print("\n=== FVD (GT-aligned crops) ===")
        fvd = compute_fvd_aligned(pairs, cropper, args.i3d_path, args.device)
        if fvd is not None:
            results['FVD'] = fvd
            print(f"FVD: {fvd:.4f}")
        else:
            print("FVD: could not compute (not enough valid videos)")

    # Summary
    print("\n=== Summary (GT-aligned crops) ===")
    for metric, val in results.items():
        print(f"  {metric}: {val:.4f}")

    with open(os.path.join(args.output_dir, 'metrics_aligned.log'), 'w') as f:
        for metric, val in results.items():
            f.write(f"{metric}: {val:.6f}\n")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
