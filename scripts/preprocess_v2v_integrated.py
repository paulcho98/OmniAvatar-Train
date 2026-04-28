"""Integrated V2V data preprocessing: directory setup + VAE + audio + ref latents.

Creates the full training data structure from raw videos + metadata CSV.
Skips directories that already exist with all required files.

Pipeline:
  Phase 1 (CPU): Create directories, symlink videos, extract audio, write prompts,
                 symlink text embeddings.
  Phase 2 (GPU): Multi-GPU VAE encoding (GT + masked + ref segment) and Wav2Vec2
                 audio encoding. Threaded prefetch for I/O overlap.

Output per new video directory:
    {video_stem}/
        sub_clip.mp4              # symlink to source
        audio.wav                 # 16kHz mono PCM (ffmpeg)
        prompt.txt                # text prompt
        text_emb.pt               # symlink to common_prompt.pt
        vae_latents_mask_all.pt   # {input_latents, masked_latents} [16,21,64,64] bf16
        audio_emb_omniavatar.pt   # {audio_emb: [num_frames, 10752], metadata}
        ref_latents.pt            # {ref_sequence_latents: [16,21,64,64], metadata}

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/preprocess_v2v_integrated.py \\
        --csv_path /path/to/metadata.csv \\
        --video_dir /path/to/raw_videos/ \\
        --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \\
        --wav2vec_path pretrained_models/wav2vec2-base-960h \\
        --latentsync_mask_path /path/to/mask.png \\
        --output_dir /home/work/stableavatar_data/v2v_training_data/ \\
        --batch_size 4
"""

import argparse
import csv
import gc
import logging
import os
import queue
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import cv2
import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.models.wav2vec import Wav2VecModel
from OmniAvatar.wan_video import WanVideoPipeline

logger = logging.getLogger(__name__)

# Files required for a directory to be considered fully processed
REQUIRED_FILES = [
    "sub_clip.mp4", "audio.wav", "prompt.txt", "text_emb.pt",
    "vae_latents_mask_all.pt", "audio_emb_omniavatar.pt", "ref_latents.pt",
]


# ---------------------------------------------------------------------------
# Phase 1: CPU directory setup
# ---------------------------------------------------------------------------

def setup_directory(row, video_dir, common_prompt_path, output_dir, force=False):
    """Create directory structure for a single video from CSV row.

    Returns (out_dir, status) where status is one of:
    - "SKIP_COMPLETE": directory exists with all required files
    - "NEW": new directory created, needs GPU processing
    - "PARTIAL": directory exists but missing GPU files, needs GPU processing
    - "SKIP_MISSING": source video not found
    - "ERROR: ...": error message
    """
    video_filename = row["video"]
    prompt = row.get("prompt", "a person is talking") or "a person is talking"
    video_stem = video_filename.replace(".mp4", "")

    out_dir = os.path.join(output_dir, video_stem)

    # Check if fully complete
    if not force and os.path.isdir(out_dir):
        if all(os.path.exists(os.path.join(out_dir, f)) for f in REQUIRED_FILES):
            return (out_dir, "SKIP_COMPLETE")

    # Check source video exists
    src_video = os.path.join(video_dir, video_filename)
    if not os.path.exists(src_video):
        return (out_dir, "SKIP_MISSING")

    os.makedirs(out_dir, exist_ok=True)

    # 1. Symlink source MP4 as sub_clip.mp4
    sub_clip_path = os.path.join(out_dir, "sub_clip.mp4")
    if not os.path.exists(sub_clip_path):
        os.symlink(os.path.abspath(src_video), sub_clip_path)

    # 2. Extract audio as WAV (16kHz mono PCM)
    audio_path = os.path.join(out_dir, "audio.wav")
    if not os.path.exists(audio_path):
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", src_video,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    audio_path,
                ],
                capture_output=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return (out_dir, "ERROR: audio extraction timeout")
        if not os.path.exists(audio_path):
            return (out_dir, "ERROR: audio extraction failed")

    # 3. Write prompt.txt
    prompt_path = os.path.join(out_dir, "prompt.txt")
    if not os.path.exists(prompt_path):
        with open(prompt_path, "w") as f:
            f.write(prompt)

    # 4. Symlink text embedding
    text_emb_path = os.path.join(out_dir, "text_emb.pt")
    if not os.path.exists(text_emb_path):
        if common_prompt_path and os.path.exists(common_prompt_path):
            os.symlink(os.path.abspath(common_prompt_path), text_emb_path)

    # Determine status
    gpu_files = ["vae_latents_mask_all.pt", "audio_emb_omniavatar.pt", "ref_latents.pt"]
    if all(os.path.exists(os.path.join(out_dir, f)) for f in gpu_files) and not force:
        return (out_dir, "SKIP_COMPLETE")

    return (out_dir, "PARTIAL" if os.path.isdir(out_dir) else "NEW")


def run_phase1_cpu(args):
    """Phase 1: Create directory structure for all videos in CSV.

    Returns list of directories that need GPU processing.
    """
    with open(args.csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info(f"CSV has {len(rows)} entries")

    new_dirs = []
    stats = {"complete": 0, "new": 0, "partial": 0, "missing": 0, "error": 0}

    with ProcessPoolExecutor(max_workers=args.num_cpu_workers) as executor:
        futures = {
            executor.submit(
                setup_directory,
                row, args.video_dir, args.common_prompt_path,
                args.output_dir, args.force,
            ): row
            for row in rows
        }

        for i, future in enumerate(as_completed(futures)):
            out_dir, status = future.result()

            if status == "SKIP_COMPLETE":
                stats["complete"] += 1
            elif status == "SKIP_MISSING":
                stats["missing"] += 1
            elif status.startswith("ERROR"):
                stats["error"] += 1
                logger.warning(f"{os.path.basename(out_dir)}: {status}")
            else:
                # NEW or PARTIAL — needs GPU processing
                stats["new" if status == "NEW" else "partial"] += 1
                new_dirs.append(out_dir)

            if (i + 1) % 5000 == 0:
                logger.info(f"  Phase 1 progress: {i + 1}/{len(rows)} "
                            f"(complete={stats['complete']}, new={stats['new']}, "
                            f"partial={stats['partial']}, missing={stats['missing']})")

    logger.info(f"Phase 1 done: {stats}")
    return sorted(new_dirs)


# ---------------------------------------------------------------------------
# Phase 2: GPU encoding
# ---------------------------------------------------------------------------

def read_video_frames(video_path, num_frames, height, width, start_frame=0):
    """Read frames from video, resize, normalize to [-1, 1].

    Returns: frames [T, H, W, 3] float32 in [-1, 1], total_frames count.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            frame = frame.astype(np.float32) / 127.5 - 1.0
            frames.append(frame)
        else:
            break
    cap.release()

    # Pad to num_frames if shorter
    if len(frames) == 0:
        frames.append(np.zeros((height, width, 3), dtype=np.float32))
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return np.stack(frames), total_frames


def encode_audio(wav2vec, feature_extractor, audio_np, num_video_frames, device):
    """Encode audio → [1, num_video_frames, 10752].

    Matches precompute_audio_omniavatar.py:67-90 exactly.
    """
    input_values = np.squeeze(
        feature_extractor(audio_np, sampling_rate=16000).input_values
    )
    input_values = torch.from_numpy(input_values).float().to(device=device)
    input_values = input_values.unsqueeze(0)

    # Pad to match video length
    samples_per_frame = feature_extractor.sampling_rate // 25  # 640
    target_samples = num_video_frames * samples_per_frame
    if input_values.shape[1] < target_samples:
        input_values = F.pad(input_values, (0, target_samples - input_values.shape[1]))

    with torch.no_grad():
        hidden_states = wav2vec(
            input_values, seq_len=num_video_frames, output_hidden_states=True,
        )
        audio_emb = hidden_states.last_hidden_state
        for hs in hidden_states.hidden_states:
            audio_emb = torch.cat((audio_emb, hs), -1)

    return audio_emb  # [1, num_video_frames, 10752]


def load_and_prepare_video(video_dir, mask_binary, args):
    """Load and prepare one video for GPU encoding. CPU-only, thread-safe.

    Returns (video_dir, gt_tensor, masked_tensor, ref_tensor, audio_np, metadata)
    or (video_dir, None, error_reason).
    """
    video_path = os.path.join(video_dir, "sub_clip.mp4")
    audio_path = os.path.join(video_dir, "audio.wav")

    if not os.path.exists(video_path):
        return (video_dir, None, "MISSING_VIDEO")

    # Check which GPU files are missing
    has_vae = os.path.exists(os.path.join(video_dir, "vae_latents_mask_all.pt"))
    has_audio = os.path.exists(os.path.join(video_dir, "audio_emb_omniavatar.pt"))
    has_ref = os.path.exists(os.path.join(video_dir, "ref_latents.pt"))

    if has_vae and has_audio and has_ref and not args.force:
        return (video_dir, None, "SKIP_COMPLETE")

    try:
        num_frames = args.num_frames
        height = args.height
        width = args.width

        # GT frames
        gt_frames, total_frames = read_video_frames(
            video_path, num_frames, height, width, start_frame=0
        )
        gt_tensor = torch.from_numpy(gt_frames).permute(3, 0, 1, 2)  # [3, T, H, W]

        # Masked frames
        masked_tensor = gt_tensor.clone()
        mask_2d = mask_binary.view(1, 1, height, width)
        if args.legacy_twostep_mask:
            # Legacy: leave frame 0 unmasked, mask frames 1+ only.
            # Frame 0 is re-encoded separately in process_batch.
            masked_tensor[:, 1:] *= mask_2d
        else:
            masked_tensor *= mask_2d

        # Reference segment
        if total_frames >= 2 * num_frames:
            ref_start = num_frames
        else:
            ref_start = max(0, total_frames - num_frames)

        ref_frames, _ = read_video_frames(
            video_path, num_frames, height, width, start_frame=ref_start
        )
        ref_tensor = torch.from_numpy(ref_frames).permute(3, 0, 1, 2)  # [3, T, H, W]

        # Audio — load full audio (not trimmed to num_frames), encode with
        # total_frames to match existing precompute_audio_omniavatar.py behavior.
        # Training script slices [:num_frames] at load time.
        audio_np = None
        if not has_audio:
            if os.path.exists(audio_path):
                audio_np, _ = librosa.load(audio_path, sr=16000)

        metadata = {
            "total_frames": total_frames,
            "ref_start": ref_start,
            "num_frames": num_frames,
            "height": height,
            "width": width,
        }

        # Mark which files we need to produce
        metadata["need_vae"] = not has_vae or args.force
        metadata["need_audio"] = not has_audio or args.force
        metadata["need_ref"] = not has_ref or args.force

        return (video_dir, gt_tensor, masked_tensor, ref_tensor, audio_np, metadata)

    except Exception as e:
        return (video_dir, None, str(e))


def batch_vae_encode(vae, video_batch, device, dtype):
    """Encode a batch of videos through VAE. Falls back to single on OOM.

    Input:  [B, 3, T, H, W]
    Output: [B, 16, T_lat, H_lat, W_lat]
    """
    try:
        video_batch = video_batch.to(device=device, dtype=dtype)
        with torch.no_grad():
            latents = vae.encode(video_batch, device=device, tiled=False)
        return latents
    except torch.cuda.OutOfMemoryError:
        logger.warning(f"OOM with batch {video_batch.shape[0]}, falling back to single")
        torch.cuda.empty_cache()
        results = []
        for i in range(video_batch.shape[0]):
            single = video_batch[i:i + 1].to(device=device, dtype=dtype)
            with torch.no_grad():
                latents = vae.encode(single, device=device, tiled=False)
            results.append(latents)
        return torch.cat(results, dim=0)


def _prefetch_batches(worker_dirs, mask_binary, args, wlog, prefetch_queue, stats,
                      num_loader_threads=4):
    """Background thread: loads and preprocesses video batches."""
    batch_size = args.batch_size

    with ThreadPoolExecutor(max_workers=num_loader_threads) as pool:
        for batch_start in range(0, len(worker_dirs), batch_size):
            batch_dirs = worker_dirs[batch_start:batch_start + batch_size]

            futures = {
                pool.submit(load_and_prepare_video, vdir, mask_binary, args): vdir
                for vdir in batch_dirs
            }

            valid_items = []
            for future in as_completed(futures):
                result = future.result()
                vdir = result[0]
                if result[1] is None:
                    reason = result[2]
                    if reason == "SKIP_COMPLETE":
                        stats["skipped"] += 1
                    elif reason == "MISSING_VIDEO":
                        stats["errors"] += 1
                        wlog.warning(f"Missing video: {vdir}")
                    else:
                        stats["errors"] += 1
                        wlog.error(f"Error loading {vdir}: {reason}")
                else:
                    valid_items.append(result)

            if valid_items:
                prefetch_queue.put(valid_items)

    prefetch_queue.put(None)  # Sentinel


def process_batch(valid_items, vae, wav2vec, feature_extractor, device, dtype, args):
    """Process a batch of videos: VAE encode (GT + masked + ref) + audio encode.

    Returns list of (video_dir, save_dict) for each video.
    """
    results = []

    # Separate items by what they need
    need_vae_items = [item for item in valid_items if item[5]["need_vae"]]
    need_ref_items = [item for item in valid_items if item[5]["need_ref"]]

    # --- VAE encode GT + masked ---
    if need_vae_items:
        gt_batch = torch.stack([item[1] for item in need_vae_items])       # [B, 3, T, H, W]
        masked_batch = torch.stack([item[2] for item in need_vae_items])   # [B, 3, T, H, W]

        input_latents_batch = batch_vae_encode(vae, gt_batch, device, dtype)
        masked_latents_batch = batch_vae_encode(vae, masked_batch, device, dtype)

        if args.legacy_twostep_mask:
            # Re-encode frame 0 with masking applied, then splice into position 0.
            # masked_batch has frame 0 unmasked; we need it masked for the splice.
            mask_2d = torch.from_numpy(
                cv2.resize(
                    (cv2.imread(args.latentsync_mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0),
                    (args.width, args.height), interpolation=cv2.INTER_LINEAR,
                )
            ).gt(0.5).float()
            frame0_batch = gt_batch[:, :, :1] * mask_2d[None, None, None, :, :]  # [B, 3, 1, H, W]
            frame0_latents = batch_vae_encode(vae, frame0_batch, device, dtype)   # [B, 16, 1, H, W]
            masked_latents_batch[:, :, :1] = frame0_latents
    else:
        input_latents_batch = None
        masked_latents_batch = None

    # --- VAE encode ref segments ---
    if need_ref_items:
        ref_batch = torch.stack([item[3] for item in need_ref_items])      # [B, 3, T, H, W]
        ref_latents_batch = batch_vae_encode(vae, ref_batch, device, dtype)
    else:
        ref_latents_batch = None

    # --- Build per-video results ---
    vae_idx = 0
    ref_idx = 0

    for item in valid_items:
        video_dir = item[0]
        audio_np = item[4]
        metadata = item[5]

        save_dict = {"video_dir": video_dir}

        # VAE latents
        if metadata["need_vae"]:
            save_dict["input_latents"] = input_latents_batch[vae_idx].cpu()
            save_dict["masked_latents"] = masked_latents_batch[vae_idx].cpu()
            vae_idx += 1

        # Ref latents
        if metadata["need_ref"]:
            save_dict["ref_latents"] = ref_latents_batch[ref_idx].cpu()
            save_dict["ref_metadata"] = {
                "ref_start": metadata["ref_start"],
                "total_frames": metadata["total_frames"],
                "num_frames": metadata["num_frames"],
                "height": metadata["height"],
                "width": metadata["width"],
            }
            ref_idx += 1

        # Audio
        if metadata["need_audio"] and audio_np is not None:
            try:
                audio_emb = encode_audio(
                    wav2vec, feature_extractor, audio_np,
                    metadata["total_frames"], device,
                )
                save_dict["audio_emb"] = audio_emb.squeeze(0).cpu()  # [total_frames, 10752]
                save_dict["audio_metadata"] = {
                    "total_video_frames": metadata["total_frames"],
                    "fps": 25,
                    "audio_length_samples": len(audio_np),
                    "feature_dim": audio_emb.shape[-1],
                }
            except Exception as e:
                logger.error(f"Audio encode error for {video_dir}: {e}")

        results.append(save_dict)

    return results


def save_results(save_dict):
    """Save precomputed files for a single video. Thread-safe."""
    video_dir = save_dict["video_dir"]

    # vae_latents_mask_all.pt
    if "input_latents" in save_dict:
        torch.save({
            "input_latents": save_dict["input_latents"],
            "masked_latents": save_dict["masked_latents"],
        }, os.path.join(video_dir, "vae_latents_mask_all.pt"))

    # ref_latents.pt
    if "ref_latents" in save_dict:
        torch.save({
            "ref_sequence_latents": save_dict["ref_latents"],
            "metadata": save_dict["ref_metadata"],
        }, os.path.join(video_dir, "ref_latents.pt"))

    # audio_emb_omniavatar.pt
    if "audio_emb" in save_dict:
        torch.save({
            "audio_emb": save_dict["audio_emb"],
            "metadata": save_dict["audio_metadata"],
        }, os.path.join(video_dir, "audio_emb_omniavatar.pt"))


def worker_process(gpu_id, worker_dirs, args):
    """GPU worker: load VAE + Wav2Vec2, process videos with prefetch pipeline."""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    torch.set_num_threads(4)

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    logging.basicConfig(
        level=logging.INFO,
        format=f"[GPU {gpu_id}] %(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    wlog = logging.getLogger(f"worker_{gpu_id}")

    if not worker_dirs:
        wlog.info("No videos assigned, exiting")
        return

    batch_size = args.batch_size
    num_batches = (len(worker_dirs) + batch_size - 1) // batch_size
    wlog.info(f"Assigned {len(worker_dirs)} videos (batch_size={batch_size}, ~{num_batches} batches)")

    # Load VAE
    wlog.info(f"Loading VAE from {args.vae_path}...")
    model_manager = ModelManager(device="cpu", infer=True)
    model_manager.load_model(args.vae_path, device="cpu", torch_dtype=dtype)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=dtype, device="cpu")
    vae = pipe.vae
    vae.to(device=device)
    wlog.info("VAE loaded")

    # Load Wav2Vec2
    wlog.info(f"Loading Wav2Vec2 from {args.wav2vec_path}...")
    wav2vec = Wav2VecModel.from_pretrained(args.wav2vec_path, local_files_only=True)
    wav2vec.eval()
    wav2vec.requires_grad_(False)
    wav2vec.feature_extractor._freeze_parameters()
    wav2vec.to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)
    wlog.info("Wav2Vec2 loaded")

    # Load and binarize LatentSync mask
    mask_img = Image.open(args.latentsync_mask_path).convert("L")
    mask_np = np.array(mask_img).astype(np.float32) / 255.0
    if mask_np.shape[0] != args.height or mask_np.shape[1] != args.width:
        mask_np = cv2.resize(mask_np, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
    mask_binary = torch.from_numpy((mask_np > 0.5).astype(np.float32))

    # Stats
    stats = {"skipped": 0, "errors": 0}
    processed = 0

    # Prefetch pipeline
    prefetch_q = queue.Queue(maxsize=2)
    loader_thread = threading.Thread(
        target=_prefetch_batches,
        args=(worker_dirs, mask_binary, args, wlog, prefetch_q, stats,
              args.num_loader_threads),
        daemon=True,
    )
    loader_thread.start()

    # Save thread pool
    save_pool = ThreadPoolExecutor(max_workers=4)
    save_futures = []

    pbar = tqdm(total=num_batches, desc=f"GPU {gpu_id}", position=gpu_id)

    while True:
        batch_data = prefetch_q.get()
        if batch_data is None:
            break

        try:
            results = process_batch(
                batch_data, vae, wav2vec, feature_extractor, device, dtype, args
            )
        except Exception as e:
            wlog.error(f"Error processing batch: {e}")
            traceback.print_exc()
            stats["errors"] += len(batch_data)
            torch.cuda.empty_cache()
            pbar.update(1)
            continue

        # Save in background
        for save_dict in results:
            fut = save_pool.submit(save_results, save_dict)
            save_futures.append(fut)
            processed += 1

        pbar.update(1)
        pbar.set_postfix(done=processed, skip=stats["skipped"], err=stats["errors"])

        # Periodic memory cleanup
        if processed % 100 == 0:
            torch.cuda.empty_cache()

    # Wait for all saves
    for fut in save_futures:
        try:
            fut.result()
        except Exception as e:
            wlog.error(f"Save error: {e}")

    loader_thread.join()
    save_pool.shutdown()
    pbar.close()

    wlog.info(f"Done: processed={processed}, skipped={stats['skipped']}, errors={stats['errors']}")

    del vae, wav2vec
    torch.cuda.empty_cache()
    gc.collect()


def run_phase2_gpu(new_dirs, args):
    """Phase 2: Multi-GPU encoding of new video directories."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")
    logger.info(f"Found {num_gpus} GPUs")

    # Split directories across GPUs
    splits = []
    chunk_size = len(new_dirs) // num_gpus
    remainder = len(new_dirs) % num_gpus
    start = 0
    for i in range(num_gpus):
        end = start + chunk_size + (1 if i < remainder else 0)
        splits.append(new_dirs[start:end])
        start = end

    # Launch workers
    ctx = mp.get_context("spawn")
    processes = []
    for gpu_id in range(num_gpus):
        p = ctx.Process(target=worker_process, args=(gpu_id, splits[gpu_id], args))
        p.start()
        processes.append(p)
        logger.info(f"Started worker on GPU {gpu_id} with {len(splits[gpu_id])} videos")

    for p in processes:
        p.join()

    logger.info("All GPU workers finished.")


# ---------------------------------------------------------------------------
# Path list generation
# ---------------------------------------------------------------------------

def generate_path_list(output_dir):
    """Scan output_dir for all complete video directories and write path list."""
    all_dirs = []
    for entry in os.scandir(output_dir):
        if entry.is_dir():
            if os.path.exists(os.path.join(entry.path, "sub_clip.mp4")):
                all_dirs.append(os.path.abspath(entry.path))

    all_dirs.sort()

    list_path = os.path.join(output_dir, "video_square_path_combined.txt")
    with open(list_path, "w") as f:
        for d in all_dirs:
            f.write(d + "\n")

    # Count complete vs incomplete
    complete = 0
    incomplete = 0
    for d in all_dirs:
        if all(os.path.exists(os.path.join(d, f)) for f in REQUIRED_FILES):
            complete += 1
        else:
            incomplete += 1

    logger.info(f"Path list written to: {list_path}")
    logger.info(f"Total directories: {len(all_dirs)} (complete={complete}, incomplete={incomplete})")
    return list_path


def validate_outputs(new_dirs, args):
    """Validate that all new directories have the expected files."""
    ok = 0
    missing = []

    for d in new_dirs:
        missing_files = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(d, f))]
        if missing_files:
            missing.append((d, missing_files))
        else:
            ok += 1

    logger.info(f"Validation: {ok}/{len(new_dirs)} directories complete")
    if missing:
        logger.warning(f"{len(missing)} directories have missing files:")
        for d, files in missing[:10]:
            logger.warning(f"  {os.path.basename(d)}: missing {files}")
        if len(missing) > 10:
            logger.warning(f"  ... and {len(missing) - 10} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Integrated V2V data preprocessing: directory setup + VAE + audio + ref latents"
    )
    # Input
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Metadata CSV (columns: video, prompt)")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Flat directory of source MP4 files")

    # Models
    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path to Wan2.1_VAE.pth")
    parser.add_argument("--wav2vec_path", type=str, required=True,
                        help="Path to wav2vec2-base-960h directory")
    parser.add_argument("--latentsync_mask_path", type=str, required=True,
                        help="Path to LatentSync mask PNG")

    # Text embedding
    parser.add_argument("--common_prompt_path", type=str,
                        default="/home/work/.local/combined_data/text_emb/common_prompt.pt",
                        help="Path to shared common_prompt.pt for text_emb.pt symlinks")

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed data (same as existing training data)")

    # Video params
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)

    # Processing params
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Videos per GPU batch for VAE encoding")
    parser.add_argument("--num_loader_threads", type=int, default=4,
                        help="CPU threads per GPU worker for video loading")
    parser.add_argument("--num_cpu_workers", type=int, default=16,
                        help="Parallel workers for Phase 1 CPU directory setup")

    # Flags
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if all files exist")
    parser.add_argument("--legacy_twostep_mask", action="store_true",
                        help="Replicate the original two-step masking pipeline: encode "
                             "with frame 0 unmasked, then re-encode frame 0 separately "
                             "with masking and splice it in. Produces identical latents "
                             "to the precompute_vae_latents_nomask + masked scripts.")
    parser.add_argument("--skip_gpu", action="store_true",
                        help="Only do Phase 1 (directory setup), skip GPU encoding")
    parser.add_argument("--also_save_vae_latents", action="store_true",
                        help="Also save vae_latents.pt with frame-0-unmasked masked_latents")

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[Main] %(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info(f"CSV: {args.csv_path}")
    logger.info(f"Video dir: {args.video_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    # Phase 1: CPU directory setup
    logger.info("=== Phase 1: Directory Setup ===")
    new_dirs = run_phase1_cpu(args)
    logger.info(f"Directories needing GPU processing: {len(new_dirs)}")

    if args.skip_gpu:
        logger.info("--skip_gpu: skipping GPU encoding")
        generate_path_list(args.output_dir)
        return

    if not new_dirs:
        logger.info("No new videos to process.")
        generate_path_list(args.output_dir)
        return

    # Phase 2: GPU encoding
    logger.info(f"=== Phase 2: GPU Encoding ===")
    run_phase2_gpu(new_dirs, args)

    # Generate combined path list
    generate_path_list(args.output_dir)

    # Validation
    validate_outputs(new_dirs, args)


if __name__ == "__main__":
    main()
