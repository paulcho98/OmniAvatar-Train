"""Precompute OmniAvatar-format audio embeddings for V2V training data (v2).

Fixes audio-video temporal alignment bug from v1: when audio is longer than
video, v1 compressed all audio into video_frames via linear_interpolation,
breaking the frame-to-time mapping (frame k should = audio at k*40ms).

This version uses seq_len = max(audio_natural_frames, video_frames):
- Audio shorter than video: pad with silence, seq_len = video_frames (same as v1)
- Audio longer than video: seq_len = audio_natural_frames (preserves all audio,
  correct temporal alignment). Matches OmniAvatar inference behavior.

Usage:
    # Re-process only mismatched samples (audio > video)
    CUDA_VISIBLE_DEVICES=0 python scripts/precompute_audio_omniavatar_v2.py \
        --wav2vec_path pretrained_models/wav2vec2-base-960h \
        --data_list_path /path/to/video_square_path.txt \
        --only-mismatched

    # Force re-process all samples
    CUDA_VISIBLE_DEVICES=0 python scripts/precompute_audio_omniavatar_v2.py \
        --wav2vec_path pretrained_models/wav2vec2-base-960h \
        --data_list_path /path/to/video_square_path.txt \
        --force

    # 2 GPUs in parallel, only mismatched
    CUDA_VISIBLE_DEVICES=0 python scripts/precompute_audio_omniavatar_v2.py \
        --wav2vec_path pretrained_models/wav2vec2-base-960h \
        --data_list_path /path/to/video_square_path.txt \
        --shard_id 0 --num_shards 2 --only-mismatched &
    CUDA_VISIBLE_DEVICES=1 python scripts/precompute_audio_omniavatar_v2.py \
        --wav2vec_path pretrained_models/wav2vec2-base-960h \
        --data_list_path /path/to/video_square_path.txt \
        --shard_id 1 --num_shards 2 --only-mismatched &
    wait
"""

import math
import os
import sys
import argparse
import json
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OmniAvatar.models.wav2vec import Wav2VecModel


def get_video_frame_count(video_path):
    """Get total frame count from video file via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            capture_output=True, text=True, check=True,
        )
        info = json.loads(result.stdout)
        for stream in info["streams"]:
            if stream["codec_type"] == "video":
                nb = stream.get("nb_frames")
                if nb and nb != "N/A":
                    return int(nb)
        # Fallback: duration * fps
        for stream in info["streams"]:
            if stream["codec_type"] == "video":
                r_frame_rate = stream.get("r_frame_rate", "25/1")
                num, den = map(int, r_frame_rate.split("/"))
                fps = num / den
                duration = float(stream.get("duration", 0))
                return int(duration * fps)
    except Exception:
        pass
    return None


def encode_audio(wav2vec, feature_extractor, audio_np, num_video_frames, device):
    """Encode audio → [1, seq_len, 10752].

    Uses seq_len = max(audio_natural_frames, num_video_frames) to ensure:
    - Correct temporal alignment (frame k = audio at k * 40ms)
    - At least num_video_frames embeddings (pad with silence if audio is shorter)
    - All audio content preserved (no truncation if audio is longer)

    Returns:
        audio_emb: (1, seq_len, 10752)
        seq_len: int, the actual seq_len used
        audio_natural_frames: int, frames derived from audio duration
    """
    input_values = np.squeeze(
        feature_extractor(audio_np, sampling_rate=16000).input_values
    )
    input_values = torch.from_numpy(input_values).float().to(device=device)
    input_values = input_values.unsqueeze(0)

    samples_per_frame = feature_extractor.sampling_rate // 25  # 640
    audio_natural_frames = math.ceil(input_values.shape[1] / samples_per_frame)
    seq_len = max(audio_natural_frames, num_video_frames)
    target_samples = seq_len * samples_per_frame

    # Pad to frame boundary (audio shorter than target) or trim excess sub-frame samples
    if input_values.shape[1] < target_samples:
        input_values = F.pad(input_values, (0, target_samples - input_values.shape[1]))
    elif input_values.shape[1] > target_samples:
        input_values = input_values[:, :target_samples]

    with torch.no_grad():
        hidden_states = wav2vec(
            input_values, seq_len=seq_len, output_hidden_states=True,
        )
        # Concatenate: last_hidden_state + 13 hidden_states = 14 × 768 = 10752
        audio_emb = hidden_states.last_hidden_state
        for hs in hidden_states.hidden_states:
            audio_emb = torch.cat((audio_emb, hs), -1)

    return audio_emb, seq_len, audio_natural_frames  # (1, seq_len, 10752)


def process_directory(video_dir, wav2vec, feature_extractor, device, output_name,
                      force=False, only_mismatched=False):
    """Process a single video directory."""
    output_path = os.path.join(video_dir, output_name)

    # Skip if already exists and not forcing
    if not force and not only_mismatched and os.path.exists(output_path):
        return "skipped"

    audio_path = os.path.join(video_dir, "audio.wav")
    if not os.path.exists(audio_path):
        return "missing audio.wav"

    # Get total video frame count from metadata in vae_latents.pt or from video file
    total_frames = None
    vae_path = os.path.join(video_dir, "vae_latents.pt")
    if os.path.exists(vae_path):
        try:
            vae_data = torch.load(vae_path, map_location="cpu", weights_only=False)
            meta = vae_data.get("metadata", {})
            total_frames = meta.get("video_length") or meta.get("clip_length")
        except Exception:
            pass

    if total_frames is None:
        video_path = os.path.join(video_dir, "sub_clip.mp4")
        if os.path.exists(video_path):
            total_frames = get_video_frame_count(video_path)

    if total_frames is None:
        return "cannot determine frame count"

    # --only-mismatched: check existing embedding, skip if audio <= video
    if only_mismatched and os.path.exists(output_path):
        try:
            existing = torch.load(output_path, map_location="cpu", weights_only=False)
            existing_meta = existing.get("metadata", {})
            al = existing_meta.get("audio_length_samples", 0)
            vf = existing_meta.get("total_video_frames", 0)
            if al <= vf * 640:
                return "skipped (audio <= video)"
        except Exception:
            pass  # If can't read existing, re-process

    # Load audio
    try:
        audio_np, _ = librosa.load(audio_path, sr=16000)
    except Exception as e:
        return f"audio load error: {e}"

    # Encode
    try:
        audio_emb, seq_len, audio_natural_frames = encode_audio(
            wav2vec, feature_extractor, audio_np, total_frames, device
        )
    except Exception as e:
        return f"encode error: {e}"

    # Save
    torch.save({
        "audio_emb": audio_emb.squeeze(0).cpu(),  # [seq_len, 10752]
        "metadata": {
            "total_video_frames": total_frames,
            "seq_len": seq_len,
            "audio_natural_frames": audio_natural_frames,
            "fps": 25,
            "audio_length_samples": len(audio_np),
            "feature_dim": audio_emb.shape[-1],
        },
    }, output_path)

    return "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Precompute OmniAvatar audio embeddings (v2 — hybrid seq_len fix)"
    )
    parser.add_argument("--wav2vec_path", type=str, required=True)
    parser.add_argument("--data_list_path", type=str, required=True,
                        help="Text file with one video directory per line")
    parser.add_argument("--output_name", type=str, default="audio_emb_omniavatar.pt",
                        help="Output filename in each video directory")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--only-mismatched", action="store_true",
                        help="Only re-process samples where audio > video duration")
    args = parser.parse_args()

    # Load video directories
    with open(args.data_list_path) as f:
        all_dirs = [line.strip() for line in f if line.strip()]

    # Shard
    shard_dirs = all_dirs[args.shard_id::args.num_shards]
    print(f"[Shard {args.shard_id}/{args.num_shards}] Processing {len(shard_dirs)}/{len(all_dirs)} directories")
    if args.only_mismatched:
        print("[Mode] Only re-processing samples where audio > video duration")

    # Load model
    print(f"Loading Wav2Vec2 from {args.wav2vec_path}...")
    wav2vec = Wav2VecModel.from_pretrained(args.wav2vec_path, local_files_only=True)
    wav2vec.eval()
    wav2vec.requires_grad_(False)
    wav2vec.feature_extractor._freeze_parameters()
    wav2vec.to(args.device)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)

    # Process
    stats = {"ok": 0, "skipped": 0, "error": 0}
    errors = []

    for video_dir in tqdm(shard_dirs, desc=f"Shard {args.shard_id}"):
        if not os.path.isdir(video_dir):
            stats["error"] += 1
            errors.append(f"{video_dir}: not a directory")
            continue

        result = process_directory(
            video_dir, wav2vec, feature_extractor, args.device, args.output_name,
            force=args.force, only_mismatched=args.only_mismatched,
        )

        if result == "ok":
            stats["ok"] += 1
        elif result.startswith("skipped"):
            stats["skipped"] += 1
        else:
            stats["error"] += 1
            errors.append(f"{os.path.basename(video_dir)}: {result}")

    print(f"\n[Shard {args.shard_id}] Done: {stats['ok']} ok, {stats['skipped']} skipped, {stats['error']} errors")
    if errors:
        print("Errors:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()
