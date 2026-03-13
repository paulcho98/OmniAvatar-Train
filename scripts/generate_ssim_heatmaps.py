#!/usr/bin/env python3
"""Generate per-pixel SSIM heatmaps for face-cropped evaluation regions.

Uses the same MediaPipe face detection + 224x224 crop + grayscale pipeline
as eval_ssim_lmd.py, but with full=True to get per-pixel SSIM maps.
Saves per-sample average heatmaps and a summary grid.
"""

import argparse
import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from skimage.metrics import structural_similarity as ssim_f
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


class FaceCropper:
    """Matches eval_ssim_lmd.py face cropping exactly."""

    def __init__(self, resolution=(224, 224), min_detection_confidence=0.5,
                 fallback_detection_confidence=0.2):
        self.resolution = resolution
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )
        self.face_detector_fallback = None
        if fallback_detection_confidence < min_detection_confidence:
            self.face_detector_fallback = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=fallback_detection_confidence,
            )

    def detect_face(self, image_bgr, use_fallback=False):
        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if use_fallback and self.face_detector_fallback is not None:
            results = self.face_detector_fallback.process(image_rgb)
        else:
            results = self.face_detector.process(image_rgb)
        if not results.detections:
            raise RuntimeError("Face not detected")
        detection = results.detections[0]
        bb = detection.location_data.relative_bounding_box
        xmin = max(0, int(bb.xmin * width))
        ymin = max(0, int(bb.ymin * height))
        xmax = min(width, xmin + int(bb.width * width))
        ymax = min(height, ymin + int(bb.height * height))
        return image_bgr[ymin:ymax, xmin:xmax]

    def crop_and_resize(self, image_bgr):
        face = self.detect_face(image_bgr)
        return cv2.resize(face, (self.resolution[1], self.resolution[0]),
                          interpolation=cv2.INTER_AREA)

    def crop_and_resize_safe(self, image_bgr):
        """Returns (cropped_face, success). Falls back to lower confidence."""
        try:
            return self.crop_and_resize(image_bgr), True
        except RuntimeError:
            if self.face_detector_fallback is not None:
                try:
                    face = self.detect_face(image_bgr, use_fallback=True)
                    return cv2.resize(face, (self.resolution[1], self.resolution[0]),
                                      interpolation=cv2.INTER_AREA), True
                except RuntimeError:
                    pass
            return None, False


def read_video_frames(video_path, max_frames=None):
    """Read video frames as list of BGR numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def compute_ssim_heatmap(gt_gray, gen_gray):
    """Compute per-pixel SSIM map between two grayscale images."""
    score, ssim_map = ssim_f(gt_gray, gen_gray, full=True)
    return score, ssim_map


def make_heatmap_image(ssim_map, vmin=0.5, vmax=1.0):
    """Convert SSIM map to a colored heatmap image (RGB uint8)."""
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    colormap = cm.RdYlGn  # Red=low, Yellow=mid, Green=high
    colored = colormap(norm(ssim_map))
    return (colored[:, :, :3] * 255).astype(np.uint8)


def process_sample(gt_path, gen_path, face_cropper, max_frames=None):
    """Process a single video pair, return average SSIM map and per-frame scores."""
    gt_frames = read_video_frames(gt_path, max_frames)
    gen_frames = read_video_frames(gen_path, max_frames)

    n_frames = min(len(gt_frames), len(gen_frames))
    if n_frames == 0:
        return None, [], 0, 0

    ssim_maps = []
    scores = []
    skipped = 0

    for i in range(n_frames):
        gt_crop, gt_ok = face_cropper.crop_and_resize_safe(gt_frames[i])
        gen_crop, gen_ok = face_cropper.crop_and_resize_safe(gen_frames[i])

        if not gt_ok or not gen_ok:
            skipped += 1
            continue

        gt_gray = cv2.cvtColor(gt_crop, cv2.COLOR_BGR2GRAY)
        gen_gray = cv2.cvtColor(gen_crop, cv2.COLOR_BGR2GRAY)

        score, smap = compute_ssim_heatmap(gt_gray, gen_gray)
        ssim_maps.append(smap)
        scores.append(score)

    if not ssim_maps:
        return None, [], skipped, n_frames

    avg_map = np.mean(ssim_maps, axis=0)
    return avg_map, scores, skipped, n_frames


def save_comparison_figure(sample_name, maps_dict, scores_dict, output_path):
    """Save a side-by-side comparison figure for one sample across methods."""
    methods = list(maps_dict.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    vmin, vmax = 0.4, 1.0
    im = None

    for ax, method in zip(axes, methods):
        smap = maps_dict[method]
        if smap is None:
            ax.text(0.5, 0.5, 'No face\ndetected', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{method}\n(no data)')
            ax.axis('off')
            continue

        im = ax.imshow(smap, cmap='RdYlGn', vmin=vmin, vmax=vmax)
        mean_score = np.mean(scores_dict[method]) if scores_dict[method] else 0
        ax.set_title(f'{method}\nSSIM={mean_score:.4f}', fontsize=11)
        ax.axis('off')

    fig.suptitle(sample_name, fontsize=14, fontweight='bold')
    if im is not None:
        plt.colorbar(im, ax=axes, shrink=0.8, label='SSIM')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate SSIM heatmaps')
    parser.add_argument('--gt_dir', required=True, help='GT videos directory (each subdir has sub_clip.mp4)')
    parser.add_argument('--methods', nargs='+', required=True,
                        help='Method specs as name:dir:suffix:strip_cfr25 '
                             '(e.g., LatentSync:/path:_synced.mp4:yes)')
    parser.add_argument('--gt_subdir_suffix', default='_cfr25',
                        help='Suffix on GT subdirectory names to strip for base name')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--crop_resolution', type=int, nargs=2, default=[224, 224])
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--samples', nargs='*', default=None,
                        help='Specific sample names to process (default: all)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    face_cropper = FaceCropper(resolution=tuple(args.crop_resolution))

    # Parse method specs
    # suffix is appended to the base name (GT dir name minus gt_subdir_suffix)
    # strip_cfr25: if 'yes', strip _cfr25 from sample name before adding suffix
    method_configs = {}
    for spec in args.methods:
        parts = spec.split(':')
        name = parts[0]
        directory = parts[1]
        suffix = parts[2] if len(parts) > 2 else '.mp4'
        strip = (parts[3].lower() == 'yes') if len(parts) > 3 else False
        method_configs[name] = {'dir': directory, 'suffix': suffix, 'strip_cfr25': strip}

    # Get sample list from GT directory
    # GT dirs have _cfr25 suffix (e.g., RD_Radio18_000_cfr25/)
    # but generated videos use base name (e.g., RD_Radio18_000_cfr25.mp4 or _synced.mp4)
    gt_dir = Path(args.gt_dir)
    if args.samples:
        sample_names = args.samples
    else:
        sample_names = sorted([
            d.name for d in gt_dir.iterdir()
            if d.is_dir() and d.name != 'face_cache'
        ])

    print(f"Processing {len(sample_names)} samples across {len(method_configs)} methods")

    # Collect all results for summary
    all_scores = {name: [] for name in method_configs}

    for si, sample_name in enumerate(sample_names):
        print(f"\n[{si+1}/{len(sample_names)}] {sample_name}")

        gt_video = gt_dir / sample_name / 'sub_clip.mp4'
        if not gt_video.exists():
            print(f"  GT video not found: {gt_video}")
            continue

        maps_dict = {}
        scores_dict = {}

        for method_name, cfg in method_configs.items():
            # Construct generated video path
            base_name = sample_name
            if cfg['strip_cfr25'] and base_name.endswith('_cfr25'):
                base_name = base_name[:-len('_cfr25')]
            gen_video = Path(cfg['dir']) / f"{base_name}{cfg['suffix']}"
            if not gen_video.exists():
                print(f"  {method_name}: video not found: {gen_video}")
                maps_dict[method_name] = None
                scores_dict[method_name] = []
                continue

            avg_map, scores, skipped, total = process_sample(
                gt_video, gen_video, face_cropper, args.max_frames
            )

            maps_dict[method_name] = avg_map
            scores_dict[method_name] = scores
            all_scores[method_name].extend(scores)

            if scores:
                print(f"  {method_name}: SSIM={np.mean(scores):.4f} "
                      f"({len(scores)}/{total} frames, {skipped} skipped)")
            else:
                print(f"  {method_name}: no valid frames")

        # Save per-sample comparison figure
        output_path = os.path.join(args.output_dir, f'{sample_name}_heatmap.png')
        save_comparison_figure(sample_name, maps_dict, scores_dict, output_path)

        # Save individual heatmap arrays for later analysis
        for method_name, smap in maps_dict.items():
            if smap is not None:
                npy_path = os.path.join(args.output_dir, f'{sample_name}_{method_name}.npy')
                np.save(npy_path, smap)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (mean SSIM across all HDTF samples)")
    print("=" * 60)
    for method_name, scores in all_scores.items():
        if scores:
            print(f"  {method_name}: {np.mean(scores):.4f} ({len(scores)} total frames)")
        else:
            print(f"  {method_name}: no data")

    # Save summary grid of worst samples
    print(f"\nHeatmaps saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
