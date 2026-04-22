"""Decode x0_pred at all ODE steps, save videos, compute perceptual + lip metrics.

Two phases:
  Phase 1 (--phase decode): VAE-decode all steps → save mp4 videos (with audio)
  Phase 2 (--phase metrics): Compute metrics on saved videos
    - Pixel MSE, SSIM, LPIPS (mouth, upper_face, full) — reference, vs decoded GT
    - LMD (Lip Landmark Distance) — reference, dlib 68-pt landmarks on mouth
    - Sharpness (Laplacian variance) — no-reference, mouth crop
    - SyncNet Sync-C, Sync-D — no-reference (uses audio)
  Phase 3 (--merge): Merge shard CSVs and plot

Both phases support --shard_id / --num_shards for multi-GPU parallelism.

Usage:
    # Decode all (4 GPUs)
    bash scripts/run_eval_ode_perceptual_v2.sh

    # Or manually:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_ode_perceptual_v2.py \
        --phase decode \
        --traj_dir /home/work/ode_full_trajectories/14B \
        --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
        --output_dir /home/work/.local/ode_analysis/14B/perceptual_v2

    CUDA_VISIBLE_DEVICES=0 python scripts/eval_ode_perceptual_v2.py \
        --phase metrics \
        --traj_dir /home/work/ode_full_trajectories/14B \
        --mask_path /home/work/.local/Self-Forcing_LipSync_StableAvatar/diffsynth/utils/mask.png \
        --output_dir /home/work/.local/ode_analysis/14B/perceptual_v2

    python scripts/eval_ode_perceptual_v2.py --merge \
        --traj_dir /home/work/ode_full_trajectories/14B \
        --output_dir /home/work/.local/ode_analysis/14B/perceptual_v2
"""

import argparse
import csv
import json
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import dlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Audio source for muxing
AUDIO_BASE_DIR = "/home/work/stableavatar_data/v2v_validation_data/recon"

# Metrics tools paths
METRICS_ROOT = "/home/work/.local/latentsync-metrics-가짜"
SHAPE_PREDICTOR = os.path.join(METRICS_ROOT, "shape_predictor_68_face_landmarks.dat")
SYNCNET_MODEL = os.path.join(METRICS_ROOT, "checkpoints/auxiliary/syncnet_v2.model")


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def discover_samples(traj_dir):
    return sorted([
        d for d in os.listdir(traj_dir)
        if os.path.isdir(os.path.join(traj_dir, d))
        and os.path.isfile(os.path.join(traj_dir, d, "ode_schedule.json"))
    ])


def shard_list(items, shard_id, num_shards):
    if shard_id is not None and num_shards is not None:
        return items[shard_id::num_shards]
    return items


def load_schedule(traj_dir, sample_names):
    with open(os.path.join(traj_dir, sample_names[0], "ode_schedule.json")) as f:
        schedule = json.load(f)
    return schedule["t_list"], schedule["num_steps"]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Decode
# ─────────────────────────────────────────────────────────────────────────────

def decode_latents_to_numpy(vae, latents, device, vae_type="wan"):
    """Decode latents to [T, H, W, 3] uint8 numpy, VAE-type agnostic."""
    if vae_type == "wan":
        # Wan VAE: expects [1, 16, F, H, W], returns [1, 3, F, H, W] in [-1, 1]
        latents = latents.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            frames = vae.decode(latents, device=device, tiled=False)
        if frames.dim() == 5:
            frames = frames[0]
        frames = ((frames.float() + 1) / 2 * 255).clamp(0, 255).byte()
        return frames.permute(1, 2, 3, 0).cpu().numpy()

    elif vae_type == "sd":
        # SD VAE (diffusers AutoencoderKL): latents saved as [4, F, H, W] at LatentSync scaling.
        # Undo scaling, decode 2D per frame.
        from einops import rearrange
        sf = vae.config.scaling_factor
        shift = vae.config.shift_factor
        latents_5d = latents.unsqueeze(0)  # [1, 4, F, H, W]
        latents_2d = rearrange(latents_5d, "b c f h w -> (b f) c h w")
        latents_2d = latents_2d.to(device=device, dtype=next(vae.parameters()).dtype)
        latents_2d = latents_2d / sf + shift
        with torch.no_grad():
            dec = vae.decode(latents_2d).sample  # [F, 3, H, W] in [-1, 1]
        dec = ((dec.clamp(-1, 1) + 1) / 2 * 255).byte()  # uint8
        return dec.permute(0, 2, 3, 1).cpu().numpy()

    else:
        raise ValueError(f"Unknown vae_type: {vae_type}")


def save_video(frames_np, path, fps=25):
    import imageio
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()


def mux_audio(video_path, audio_path, output_path):
    """Mux audio into video, trimming to shortest."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path, "-i", audio_path,
        "-c:v", "libx264", "-crf", "18", "-c:a", "aac",
        "-shortest", output_path,
    ], capture_output=True)


def run_decode(args):
    if args.vae_type == "wan" and args.vae_path is None:
        raise ValueError("--vae_path is required when --vae_type wan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_names = shard_list(discover_samples(args.traj_dir), args.shard_id, args.num_shards)
    t_list, num_steps = load_schedule(args.traj_dir, discover_samples(args.traj_dir))

    if args.shard_id is not None:
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(sample_names)} samples")

    if args.vae_type == "wan":
        from OmniAvatar.models.model_manager import ModelManager
        print("Loading Wan VAE...")
        model_manager = ModelManager(device="cpu", infer=True)
        model_manager.load_models([args.vae_path], torch_dtype=torch.bfloat16, device="cpu")
        vae_idx = model_manager.model_name.index("wan_video_vae")
        vae = model_manager.model[vae_idx].to(device)
    elif args.vae_type == "sd":
        from diffusers import AutoencoderKL
        print("Loading SD VAE (stabilityai/sd-vae-ft-mse)...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        vae = vae.to(device).eval()
    else:
        raise ValueError(f"Unknown vae_type: {args.vae_type}")

    videos_dir = os.path.join(args.output_dir, "videos")
    total_decodes = len(sample_names) * (num_steps + 1)
    decode_count = 0
    t_start = time.time()

    for si, sample_name in enumerate(sample_names):
        print(f"\n[{si+1}/{len(sample_names)}] {sample_name}")
        sample_vid_dir = os.path.join(videos_dir, sample_name)
        os.makedirs(sample_vid_dir, exist_ok=True)

        audio_path = os.path.join(AUDIO_BASE_DIR, sample_name, "audio.wav")
        has_audio = os.path.exists(audio_path)

        # Decode GT
        gt_path = os.path.join(sample_vid_dir, "gt.mp4")
        if not os.path.exists(gt_path):
            gt_latents = torch.load(
                os.path.join(args.traj_dir, sample_name, "input_latents.pt"),
                map_location="cpu", weights_only=True,
            ).float()
            gt_frames = decode_latents_to_numpy(vae, gt_latents, device, vae_type=args.vae_type)
            save_video(gt_frames, gt_path)
            if has_audio:
                mux_audio(gt_path, audio_path, gt_path.replace(".mp4", "_audio.mp4"))
        decode_count += 1

        # Decode each step
        for step_i in range(num_steps):
            step_path = os.path.join(sample_vid_dir, f"step_{step_i:03d}.mp4")
            if os.path.exists(step_path):
                decode_count += 1
                continue

            x0 = torch.load(
                os.path.join(args.traj_dir, sample_name, f"step_{step_i:03d}_x0.pt"),
                map_location="cpu", weights_only=True,
            ).float()
            frames = decode_latents_to_numpy(vae, x0, device, vae_type=args.vae_type)
            save_video(frames, step_path)

            # Mux audio for SyncNet
            if has_audio:
                mux_audio(step_path, audio_path, step_path.replace(".mp4", "_audio.mp4"))

            decode_count += 1

            if step_i % 10 == 0 or step_i == num_steps - 1:
                elapsed = time.time() - t_start
                rate = decode_count / elapsed if elapsed > 0 else 0
                remaining = (total_decodes - decode_count) / rate if rate > 0 else 0
                print(f"  step {step_i:2d} | decoded {decode_count}/{total_decodes} | "
                      f"ETA {remaining/60:.1f} min")

    print(f"\nDecode done! {decode_count} videos in {(time.time()-t_start)/60:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Metrics
# ─────────────────────────────────────────────────────────────────────────────

def load_pixel_mask(mask_path, H=512, W=512):
    mask_img = Image.open(mask_path)
    mask_arr = np.array(mask_img, dtype=np.float32)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    mask_arr = mask_arr / 255.0
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_t, size=(H, W), mode="bilinear", align_corners=False)
    return mask_resized.squeeze() > 0.5


def get_mouth_bbox(mask_keep, pad=4):
    mouth = ~mask_keep
    ys, xs = torch.where(mouth)
    H, W = mask_keep.shape
    return (
        max(0, ys.min().item() - pad),
        min(H, ys.max().item() + 1 + pad),
        max(0, xs.min().item() - pad),
        min(W, xs.max().item() + 1 + pad),
    )


def read_video_frames(path):
    """Read video → list of BGR numpy frames."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def frames_to_tensor(frames_bgr):
    """BGR uint8 list → [T, 3, H, W] float32 in [0, 1] (RGB)."""
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    arr = np.stack(rgb).astype(np.float32) / 255.0  # [T, H, W, 3]
    return torch.from_numpy(arr).permute(0, 3, 1, 2)  # [T, 3, H, W]


# -- Pixel MSE --
def compute_masked_mse(pred, gt, mask):
    mask_exp = mask.unsqueeze(0).unsqueeze(0).expand_as(pred)
    diff_sq = (pred - gt).pow(2)
    return diff_sq[mask_exp].mean().item()


# -- SSIM --
def compute_masked_ssim(pred, gt, mask):
    pred_np = pred.permute(0, 2, 3, 1).numpy()
    gt_np = gt.permute(0, 2, 3, 1).numpy()
    mask_np = mask.numpy()
    vals = []
    for t in range(pred_np.shape[0]):
        _, ssim_map = ssim_fn(
            gt_np[t], pred_np[t],
            channel_axis=2, data_range=1.0, full=True,
        )
        ssim_spatial = ssim_map.mean(axis=2)
        vals.append(ssim_spatial[mask_np].mean())
    return float(np.mean(vals))


# -- LPIPS --
def compute_lpips(lpips_model, pred, gt, bbox, device):
    y0, y1, x0, x1 = bbox
    pred_crop = pred[:, :, y0:y1, x0:x1] * 2 - 1
    gt_crop = gt[:, :, y0:y1, x0:x1] * 2 - 1
    vals = []
    for i in range(0, pred_crop.shape[0], 16):
        p = pred_crop[i:i+16].to(device)
        g = gt_crop[i:i+16].to(device)
        with torch.no_grad():
            d = lpips_model(p, g)
        vals.append(d.cpu().reshape(-1))
    return torch.cat(vals).mean().item()


def compute_lpips_full(lpips_model, pred, gt, device):
    pred_l = pred * 2 - 1
    gt_l = gt * 2 - 1
    vals = []
    for i in range(0, pred_l.shape[0], 16):
        p = pred_l[i:i+16].to(device)
        g = gt_l[i:i+16].to(device)
        with torch.no_grad():
            d = lpips_model(p, g)
        vals.append(d.cpu().reshape(-1))
    return torch.cat(vals).mean().item()


# -- Sharpness (Laplacian variance) --
def compute_mouth_sharpness(frames_bgr, bbox):
    """Laplacian variance in mouth crop, averaged over frames. No reference needed."""
    y0, y1, x0, x1 = bbox
    vals = []
    for frame in frames_bgr:
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        vals.append(lap.var())
    return float(np.mean(vals))


# -- LMD --
def compute_video_lmd(pred_frames_bgr, gt_frames_bgr, detector, predictor):
    """LMD between pred and GT videos, matching the latentsync-metrics implementation."""
    T = min(len(pred_frames_bgr), len(gt_frames_bgr))
    total_lmd = 0.0
    count = 0
    for t in range(T):
        pred_land = _extract_mouth_landmarks(pred_frames_bgr[t], detector, predictor)
        gt_land = _extract_mouth_landmarks(gt_frames_bgr[t], detector, predictor)
        if pred_land is None or gt_land is None:
            continue
        diff = pred_land - gt_land
        lmd = float(np.sum(np.linalg.norm(diff, axis=1)) / pred_land.shape[0])
        total_lmd += lmd
        count += 1
    if count == 0:
        return None
    return total_lmd / count


def _extract_mouth_landmarks(image_bgr, detector, predictor):
    """Extract 20 mouth landmarks (dlib points 48-67), mean-centered."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if not rects:
        return None
    rect = max(rects, key=lambda r: r.width() * r.height())
    shape = predictor(gray, rect)
    pts = np.asarray([(pt.x, pt.y) for pt in shape.parts()], dtype=np.float64)
    mouth = pts[48:68]
    mouth -= mouth.mean(axis=0, keepdims=True)
    return mouth


# -- SyncNet --
def compute_syncnet(video_with_audio_path, device, min_track=50):
    """Run SyncNet on a video with muxed audio. Returns (sync_d, sync_c) or None.

    min_track: minimum number of consecutive frames with a detected face for a valid
    face track. Default 50 works for 81-frame OmniAvatar videos. LatentSync produces
    16-frame videos → need min_track <= 15.
    """
    sys.path.insert(0, METRICS_ROOT)
    from eval.syncnet import SyncNetEval
    from eval.syncnet_detect import SyncNetDetector

    syncnet = SyncNetEval(device=device)
    syncnet.loadParameters(SYNCNET_MODEL)

    detect_dir = tempfile.mkdtemp(prefix="syncdet_")
    temp_dir = tempfile.mkdtemp(prefix="synctemp_")

    try:
        detector = SyncNetDetector(device=device, detect_results_dir=detect_dir)
        detector(video_path=video_with_audio_path, min_track=min_track)

        crop_dir = os.path.join(detect_dir, "crop")
        if not os.path.exists(crop_dir) or not os.listdir(crop_dir):
            return None

        sync_d_list, sync_c_list = [], []
        for video in os.listdir(crop_dir):
            vtemp = tempfile.mkdtemp(prefix="sv_", dir=temp_dir)
            try:
                _, min_dist, conf = syncnet.evaluate(
                    video_path=os.path.join(crop_dir, video), temp_dir=vtemp
                )
                sync_d_list.append(min_dist)
                sync_c_list.append(conf)
            except Exception:
                pass
            finally:
                shutil.rmtree(vtemp, ignore_errors=True)

        if not sync_d_list:
            return None
        from statistics import fmean
        return fmean(sync_d_list), fmean(sync_c_list)
    except Exception:
        return None
    finally:
        shutil.rmtree(detect_dir, ignore_errors=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_metrics(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_samples = discover_samples(args.traj_dir)
    sample_names = shard_list(all_samples, args.shard_id, args.num_shards)
    t_list, num_steps = load_schedule(args.traj_dir, all_samples)

    if args.shard_id is not None:
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(sample_names)} samples")

    videos_dir = os.path.join(args.output_dir, "videos")
    mask_keep = load_pixel_mask(args.mask_path, H=512, W=512)
    mouth_bbox = get_mouth_bbox(mask_keep)
    mouth_mask = ~mask_keep
    upper_mask = mask_keep
    full_mask = torch.ones_like(mask_keep)

    # Load LPIPS
    import lpips
    print("Loading LPIPS...")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    # Load dlib
    print("Loading dlib landmark detector...")
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

    # SyncNet: load once, reuse
    print("Loading SyncNet...")
    sys.path.insert(0, METRICS_ROOT)
    from eval.syncnet import SyncNetEval
    from eval.syncnet_detect import SyncNetDetector
    syncnet = SyncNetEval(device=str(device))
    syncnet.loadParameters(SYNCNET_MODEL)

    # CSV
    shard_suffix = f"_shard{args.shard_id}" if args.shard_id is not None else ""
    if args.sync_only:
        # Append-only sync output; filename differs so we don't clobber existing metrics.csv.
        csv_path = os.path.join(args.output_dir, f"metrics_sync_only{shard_suffix}.csv")
    else:
        csv_path = os.path.join(args.output_dir, f"metrics{shard_suffix}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "t", "sample", "metric", "region", "value"])
    sync_min_track = args.sync_min_track

    t_start = time.time()

    for si, sample_name in enumerate(sample_names):
        print(f"\n[{si+1}/{len(sample_names)}] {sample_name}")
        sample_vid_dir = os.path.join(videos_dir, sample_name)

        # Read GT video once
        gt_path = os.path.join(sample_vid_dir, "gt.mp4")
        gt_bgr = read_video_frames(gt_path)
        gt_tensor = frames_to_tensor(gt_bgr)

        # GT baselines (sharpness, SyncNet) — written with step=-1
        if not args.sync_only:
            gt_sharp = compute_mouth_sharpness(gt_bgr, mouth_bbox)
            writer.writerow([-1, "gt", sample_name, "sharpness", "mouth", f"{gt_sharp:.4f}"])

        gt_audio_path = os.path.join(sample_vid_dir, "gt_audio.mp4")
        if os.path.exists(gt_audio_path):
            detect_dir = tempfile.mkdtemp(prefix="syncdet_gt_")
            temp_dir_sync = tempfile.mkdtemp(prefix="synctemp_gt_")
            try:
                detector_sync = SyncNetDetector(
                    device=str(device), detect_results_dir=detect_dir
                )
                detector_sync(video_path=gt_audio_path, min_track=sync_min_track)
                crop_dir = os.path.join(detect_dir, "crop")
                if os.path.exists(crop_dir) and os.listdir(crop_dir):
                    sd_list, sc_list = [], []
                    for vid in os.listdir(crop_dir):
                        vtemp = tempfile.mkdtemp(prefix="sv_", dir=temp_dir_sync)
                        try:
                            _, md, cf = syncnet.evaluate(
                                video_path=os.path.join(crop_dir, vid), temp_dir=vtemp)
                            sd_list.append(md)
                            sc_list.append(cf)
                        except Exception:
                            pass
                        finally:
                            shutil.rmtree(vtemp, ignore_errors=True)
                    if sd_list:
                        from statistics import fmean
                        writer.writerow([-1, "gt", sample_name, "sync_d", "mouth", f"{fmean(sd_list):.6f}"])
                        writer.writerow([-1, "gt", sample_name, "sync_c", "mouth", f"{fmean(sc_list):.6f}"])
            except Exception as e:
                print(f"  GT SyncNet error: {e}")
            finally:
                shutil.rmtree(detect_dir, ignore_errors=True)
                shutil.rmtree(temp_dir_sync, ignore_errors=True)

        csv_file.flush()

        for step_i in range(num_steps):
            t_val = t_list[step_i]
            step_path = os.path.join(sample_vid_dir, f"step_{step_i:03d}.mp4")
            step_audio_path = step_path.replace(".mp4", "_audio.mp4")

            if not args.sync_only:
                pred_bgr = read_video_frames(step_path)
                pred_tensor = frames_to_tensor(pred_bgr)

                T = min(pred_tensor.shape[0], gt_tensor.shape[0])
                pred_t = pred_tensor[:T]
                gt_t = gt_tensor[:T]
                pred_b = pred_bgr[:T]
                gt_b = gt_bgr[:T]

                # -- Pixel MSE --
                for region, mask in [("mouth", mouth_mask), ("upper_face", upper_mask), ("full", full_mask)]:
                    v = compute_masked_mse(pred_t, gt_t, mask)
                    writer.writerow([step_i, f"{t_val:.6f}", sample_name, "pixel_mse", region, f"{v:.8f}"])

                # -- SSIM --
                for region, mask in [("mouth", mouth_mask), ("upper_face", upper_mask), ("full", full_mask)]:
                    v = compute_masked_ssim(pred_t, gt_t, mask)
                    writer.writerow([step_i, f"{t_val:.6f}", sample_name, "ssim", region, f"{v:.8f}"])

                # -- LPIPS --
                lp_mouth = compute_lpips(lpips_model, pred_t, gt_t, mouth_bbox, device)
                writer.writerow([step_i, f"{t_val:.6f}", sample_name, "lpips", "mouth", f"{lp_mouth:.8f}"])
                lp_full = compute_lpips_full(lpips_model, pred_t, gt_t, device)
                writer.writerow([step_i, f"{t_val:.6f}", sample_name, "lpips", "full", f"{lp_full:.8f}"])

                # -- Sharpness --
                sharp = compute_mouth_sharpness(pred_b, mouth_bbox)
                writer.writerow([step_i, f"{t_val:.6f}", sample_name, "sharpness", "mouth", f"{sharp:.4f}"])

                # -- LMD --
                lmd = compute_video_lmd(pred_b, gt_b, dlib_detector, dlib_predictor)
                if lmd is not None:
                    writer.writerow([step_i, f"{t_val:.6f}", sample_name, "lmd", "mouth", f"{lmd:.6f}"])

            # -- SyncNet --
            if os.path.exists(step_audio_path):
                detect_dir = tempfile.mkdtemp(prefix="syncdet_")
                temp_dir_sync = tempfile.mkdtemp(prefix="synctemp_")
                try:
                    detector_sync = SyncNetDetector(
                        device=str(device), detect_results_dir=detect_dir
                    )
                    detector_sync(video_path=step_audio_path, min_track=sync_min_track)
                    crop_dir = os.path.join(detect_dir, "crop")
                    if os.path.exists(crop_dir) and os.listdir(crop_dir):
                        sync_d_list, sync_c_list = [], []
                        for vid in os.listdir(crop_dir):
                            vtemp = tempfile.mkdtemp(prefix="sv_", dir=temp_dir_sync)
                            try:
                                _, min_dist, conf = syncnet.evaluate(
                                    video_path=os.path.join(crop_dir, vid),
                                    temp_dir=vtemp,
                                )
                                sync_d_list.append(min_dist)
                                sync_c_list.append(conf)
                            except Exception:
                                pass
                            finally:
                                shutil.rmtree(vtemp, ignore_errors=True)
                        if sync_d_list:
                            from statistics import fmean
                            writer.writerow([step_i, f"{t_val:.6f}", sample_name,
                                             "sync_d", "mouth", f"{fmean(sync_d_list):.6f}"])
                            writer.writerow([step_i, f"{t_val:.6f}", sample_name,
                                             "sync_c", "mouth", f"{fmean(sync_c_list):.6f}"])
                except Exception as e:
                    print(f"    SyncNet error step {step_i}: {e}")
                finally:
                    shutil.rmtree(detect_dir, ignore_errors=True)
                    shutil.rmtree(temp_dir_sync, ignore_errors=True)

            csv_file.flush()

            if step_i % 10 == 0 or step_i == num_steps - 1:
                elapsed = time.time() - t_start
                print(f"  step {step_i:2d} (t={t_val:.3f}) | "
                      f"elapsed {elapsed/60:.1f} min")

    csv_file.close()
    print(f"\nMetrics done! CSV: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Merge + Plot
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_plot(args):
    import pandas as pd

    shard_files = sorted(glob.glob(os.path.join(args.output_dir, "metrics_shard*.csv")))
    if shard_files:
        dfs = [pd.read_csv(f) for f in shard_files]
        df = pd.concat(dfs, ignore_index=True)
        merged_path = os.path.join(args.output_dir, "metrics.csv")
        df.to_csv(merged_path, index=False)
        print(f"Merged {len(shard_files)} shards → {merged_path} ({len(df)} rows)")
    else:
        merged_path = os.path.join(args.output_dir, "metrics.csv")
        if not os.path.exists(merged_path):
            print("No shard or merged CSV found.")
            return
        df = pd.read_csv(merged_path)

    # Load t_list
    all_samples = discover_samples(args.traj_dir)
    t_list, num_steps = load_schedule(args.traj_dir, all_samples)
    t_values = np.array(t_list[:num_steps])

    # Separate GT baselines (step=-1) from trajectory data
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    gt_rows = df[df["step"] == -1]
    gt_baselines = gt_rows.groupby(["metric", "region"])["value"].mean().reset_index()
    df_steps = df[df["step"] >= 0].copy()
    agg = df_steps.groupby(["step", "metric", "region"])["value"].mean().reset_index()

    # ── Reference metrics (vs GT): MSE, SSIM, LPIPS, LMD ──
    ref_metrics = [
        ("pixel_mse", "Pixel MSE", True, ["mouth", "upper_face", "full"]),
        ("ssim", "SSIM", False, ["mouth", "upper_face", "full"]),
        ("lpips", "LPIPS", False, ["mouth", "full"]),
        ("lmd", "LMD (lip landmarks)", False, ["mouth"]),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(28, 10))

    region_style = {
        "mouth": ("red", "o"),
        "upper_face": ("blue", "s"),
        "full": ("gray", "^"),
    }

    for col, (metric_name, title, use_log, regions) in enumerate(ref_metrics):
        ax = axes[0, col]
        sub = agg[agg["metric"] == metric_name]
        for region in regions:
            data = sub[sub["region"] == region].sort_values("step")
            if len(data) == 0:
                continue
            color, marker = region_style[region]
            ax.plot(data["step"], data["value"],
                    f"-{marker}", markersize=2, color=color,
                    label=region.replace("_", " ").title())
        ax.set_xlabel("ODE Step")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs GT")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

        # Secondary x-axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"t={t_values[i]:.2f}" for i in tick_pos])
        ax2.set_xlabel("Timestep t", fontsize=8)

        # Delta
        ax_d = axes[1, col]
        lower_better = metric_name in ("pixel_mse", "lpips", "lmd")
        for region in regions:
            data = sub[sub["region"] == region].sort_values("step")
            if len(data) == 0:
                continue
            vals = data["value"].values
            delta = np.zeros(len(vals))
            if lower_better:
                delta[1:] = vals[:-1] - vals[1:]
            else:
                delta[1:] = vals[1:] - vals[:-1]
            color, _ = region_style[region]
            offsets = {"mouth": -0.25, "upper_face": 0.0, "full": 0.25}
            ax_d.bar(data["step"].values + offsets.get(region, 0),
                     delta, width=0.25, color=color, alpha=0.7,
                     label=region.replace("_", " ").title())
        ax_d.set_xlabel("ODE Step")
        ax_d.set_ylabel(f"Δ {title}")
        ax_d.set_title(f"Per-Step Δ ({title})")
        ax_d.legend()
        ax_d.grid(True, alpha=0.3)
        ax_d.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    path1 = os.path.join(args.output_dir, "reference_metrics.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"Saved {path1}")

    # ── No-reference metrics: Sharpness, SyncNet ──
    noref_metrics = [
        ("sharpness", "Mouth Sharpness (Laplacian var)", False),
        ("sync_d", "Sync-D (lower=better sync)", False),
        ("sync_c", "Sync-C (higher=better sync)", False),
    ]

    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 5))

    for col, (metric_name, title, use_log) in enumerate(noref_metrics):
        ax = axes2[col]
        sub = agg[agg["metric"] == metric_name]
        data = sub[sub["region"] == "mouth"].sort_values("step")
        if len(data) == 0:
            ax.set_title(f"{title} (no data)")
            continue
        ax.plot(data["step"], data["value"], "r-o", markersize=3, label="Prediction")
        ax.set_xlabel("ODE Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")

        # GT baseline as horizontal dashed line
        gt_val = gt_baselines[
            (gt_baselines["metric"] == metric_name) & (gt_baselines["region"] == "mouth")
        ]["value"]
        if len(gt_val) > 0:
            ax.axhline(y=gt_val.values[0], color="green", linestyle="--",
                        linewidth=2, label=f"GT ({gt_val.values[0]:.2f})")
        ax.legend()

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_pos = [i for i in [0, 10, 20, 30, 40, 49] if i < len(t_values)]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels([f"t={t_values[i]:.2f}" for i in tick_pos])
        ax2.set_xlabel("Timestep t", fontsize=8)

    plt.tight_layout()
    path2 = os.path.join(args.output_dir, "noref_metrics.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved {path2}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, choices=["decode", "metrics"],
                        help="Which phase to run")
    parser.add_argument("--merge", action="store_true", help="Merge shards and plot")
    parser.add_argument("--traj_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default=None,
                        help="Path to Wan VAE .pth (required when --vae_type wan)")
    parser.add_argument("--vae_type", type=str, default="wan",
                        choices=["wan", "sd"],
                        help="VAE family for decoding. 'wan' = OmniAvatar Wan VAE (16ch), "
                             "'sd' = Stable Diffusion VAE via diffusers (4ch).")
    parser.add_argument("--mask_path", type=str, default=None,
                        help="LatentSync mask path (required for metrics phase)")
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--sync_min_track", type=int, default=50,
                        help="Min frames for SyncNet face track. Lower for short videos "
                             "(e.g. 15 for 16-frame LatentSync).")
    parser.add_argument("--sync_only", action="store_true",
                        help="Compute ONLY SyncNet sync_c/sync_d (skip pixel_mse/ssim/lpips/lmd/sharpness). "
                             "Writes to metrics_sync_only.csv so existing metrics.csv is preserved.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.merge:
        merge_and_plot(args)
    elif args.phase == "decode":
        run_decode(args)
    elif args.phase == "metrics":
        run_metrics(args)
    else:
        parser.error("Specify --phase decode, --phase metrics, or --merge")


if __name__ == "__main__":
    main()
