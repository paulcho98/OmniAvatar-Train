#!/usr/bin/env python3
"""Batch SyncNet evaluation for OmniAvatar validation videos.

Evaluates all videos in a directory, groups results by checkpoint step
and category (recon/mixed), and saves a summary report.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_syncnet_batch.py \
        --video_dir /home/work/.local/media/videos \
        --output /home/work/.local/media/syncnet_results.txt
"""

import argparse
import os
import sys
import tempfile
import shutil
import re
from collections import defaultdict

import torch

# SyncNet imports
sys.path.insert(0, "/home/work/.local/Self-Forcing_LipSync_StableAvatar")
from diffsynth.models.syncnet import SyncNetEval, SyncNetDetector, syncnet_eval


SYNCNET_MODEL = "/home/work/.local/LatentSync/checkpoints/auxiliary/syncnet_v2.model"
S3FD_MODEL = "/home/work/.local/LatentSync/checkpoints/auxiliary/sfd_face.pth"


def parse_video_info(filename):
    """Extract step and hash from filename like 'videos_1000_abc123.mp4'."""
    m = re.match(r"videos_(\d+)_([a-f0-9]+)\.mp4", filename)
    if m:
        return int(m.group(1)), m.group(2)
    return None, None


def evaluate_videos(video_dir, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    print("Loading SyncNet model...")
    syncnet = SyncNetEval(device=device)
    syncnet.loadParameters(SYNCNET_MODEL)

    print("Loading S3FD face detector...")
    temp_base = tempfile.mkdtemp(prefix="syncnet_eval_")
    detector = SyncNetDetector(
        device=device,
        detect_results_dir=os.path.join(temp_base, "detect"),
        s3fd_model_path=S3FD_MODEL,
    )

    # Collect videos by category
    categories = sorted([
        d for d in os.listdir(video_dir)
        if os.path.isdir(os.path.join(video_dir, d))
    ])

    # results[category][step] = [(sync_d, sync_c, offset, filename), ...]
    results = defaultdict(lambda: defaultdict(list))
    all_results = []

    for category in categories:
        cat_dir = os.path.join(video_dir, category)
        videos = sorted([f for f in os.listdir(cat_dir) if f.endswith(".mp4")])
        print(f"\n{'='*60}")
        print(f"Category: {category} ({len(videos)} videos)")
        print(f"{'='*60}")

        for video_file in videos:
            step, vid_hash = parse_video_info(video_file)
            if step is None:
                print(f"  Skipping {video_file} (unrecognized name)")
                continue

            video_path = os.path.join(cat_dir, video_file)
            eval_temp = os.path.join(temp_base, f"{category}_{step}_{vid_hash}")
            detect_dir = os.path.join(temp_base, f"detect_{category}_{step}_{vid_hash}")
            os.makedirs(eval_temp, exist_ok=True)

            # Update detector's results dir per video to avoid conflicts
            detector.detect_results_dir = detect_dir

            try:
                av_offset, sync_d, sync_c = syncnet_eval(
                    syncnet, detector, video_path, eval_temp,
                    detect_results_dir=detect_dir,
                )
                results[category][step].append((sync_d, sync_c, av_offset, video_file))
                all_results.append((category, step, sync_d, sync_c, av_offset, video_file))
                print(f"  Step {step:>5} | {video_file} | Sync-D: {sync_d:.4f} | Sync-C: {sync_c:.4f} | Offset: {av_offset}")
            except Exception as e:
                print(f"  Step {step:>5} | {video_file} | ERROR: {e}")
                results[category][step].append((None, None, None, video_file))
                all_results.append((category, step, None, None, None, video_file))

            # Clean up per-video temp dirs
            shutil.rmtree(eval_temp, ignore_errors=True)
            shutil.rmtree(detect_dir, ignore_errors=True)

    # Clean up
    shutil.rmtree(temp_base, ignore_errors=True)

    # --- Build report ---
    lines = []
    lines.append("=" * 80)
    lines.append("SyncNet Evaluation Results — OmniAvatar V2V Validation")
    lines.append("=" * 80)
    lines.append("")

    # Per-category, per-step summary
    all_steps = sorted({step for cat in results.values() for step in cat})

    for category in sorted(results.keys()):
        cat_label = category.replace("val_", "").upper()
        lines.append(f"{'─'*80}")
        lines.append(f"  {cat_label}")
        lines.append(f"{'─'*80}")
        lines.append(f"  {'Step':>6}  {'Sync-D (↓)':>12}  {'Sync-C (↑)':>12}  {'Offset':>8}  {'N':>3}")
        lines.append(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*3}")

        for step in all_steps:
            entries = results[category].get(step, [])
            valid = [(d, c, o) for d, c, o, _ in entries if d is not None]
            if valid:
                avg_d = sum(d for d, c, o in valid) / len(valid)
                avg_c = sum(c for d, c, o in valid) / len(valid)
                avg_o = sum(o for d, c, o in valid) / len(valid)
                lines.append(f"  {step:>6}  {avg_d:>12.4f}  {avg_c:>12.4f}  {avg_o:>8.1f}  {len(valid):>3}")
            else:
                lines.append(f"  {step:>6}  {'N/A':>12}  {'N/A':>12}  {'N/A':>8}  {0:>3}")

        lines.append("")

    # Cross-category summary table
    lines.append(f"{'='*80}")
    lines.append("  SUMMARY BY STEP (averaged across all videos)")
    lines.append(f"{'='*80}")
    lines.append("")
    header = f"  {'Step':>6}"
    for category in sorted(results.keys()):
        cat_label = category.replace("val_", "")
        header += f"  │  {cat_label+' D':>10}  {cat_label+' C':>10}"
    lines.append(header)
    lines.append(f"  {'─'*6}" + f"  │  {'─'*10}  {'─'*10}" * len(results))

    for step in all_steps:
        row = f"  {step:>6}"
        for category in sorted(results.keys()):
            entries = results[category].get(step, [])
            valid = [(d, c) for d, c, o, _ in entries if d is not None]
            if valid:
                avg_d = sum(d for d, c in valid) / len(valid)
                avg_c = sum(c for d, c in valid) / len(valid)
                row += f"  │  {avg_d:>10.4f}  {avg_c:>10.4f}"
            else:
                row += f"  │  {'N/A':>10}  {'N/A':>10}"
        lines.append(row)

    lines.append("")

    # Individual results
    lines.append(f"{'='*80}")
    lines.append("  INDIVIDUAL VIDEO RESULTS")
    lines.append(f"{'='*80}")
    lines.append(f"  {'Category':<12} {'Step':>5} {'Sync-D':>10} {'Sync-C':>10} {'Offset':>7}  Filename")
    lines.append(f"  {'─'*12} {'─'*5} {'─'*10} {'─'*10} {'─'*7}  {'─'*40}")

    for category, step, sync_d, sync_c, offset, filename in sorted(all_results, key=lambda x: (x[0], x[1], x[5])):
        cat_label = category.replace("val_", "")
        if sync_d is not None:
            lines.append(f"  {cat_label:<12} {step:>5} {sync_d:>10.4f} {sync_c:>10.4f} {offset:>7}  {filename}")
        else:
            lines.append(f"  {cat_label:<12} {step:>5} {'ERROR':>10} {'ERROR':>10} {'N/A':>7}  {filename}")

    lines.append("")

    report = "\n".join(lines)
    print(f"\n{report}")

    # Save to file
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing val_recon/ and val_mixed/ subdirs")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results text file")
    args = parser.parse_args()
    evaluate_videos(args.video_dir, args.output)
