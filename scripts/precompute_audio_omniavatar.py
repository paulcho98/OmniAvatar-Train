"""Precompute OmniAvatar-format audio embeddings for V2V training data.

OmniAvatar uses a custom Wav2VecModel that concatenates all 14 hidden states
(768 × 14 = 10752-dim) and applies linear_interpolation to resample to the exact
video frame count. This is incompatible with StableAvatar's [161, 768] embeddings.

This script processes each video directory's audio.wav through OmniAvatar's pipeline,
producing audio_emb_omniavatar.pt with shape [total_video_frames, 10752].

Usage:
    # 2 GPUs in parallel
    CUDA_VISIBLE_DEVICES=0 python scripts/precompute_audio_omniavatar.py \
        --wav2vec_path pretrained_models/wav2vec2-base-960h \
        --data_list_path /path/to/video_square_path.txt \
        --shard_id 0 --num_shards 2 --device cuda:0 &
    CUDA_VISIBLE_DEVICES=1 python scripts/precompute_audio_omniavatar.py \
        --wav2vec_path pretrained_models/wav2vec2-base-960h \
        --data_list_path /path/to/video_square_path.txt \
        --shard_id 1 --num_shards 2 --device cuda:0 &
    wait
"""

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
    """Encode audio → [1, num_video_frames, 10752]. Matches train.py:321-342 exactly."""
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
        # Concatenate: last_hidden_state + 13 hidden_states = 14 × 768 = 10752
        audio_emb = hidden_states.last_hidden_state
        for hs in hidden_states.hidden_states:
            audio_emb = torch.cat((audio_emb, hs), -1)

    return audio_emb  # (1, num_video_frames, 10752)


def process_directory(video_dir, wav2vec, feature_extractor, device, output_name, force=False):
    """Process a single video directory."""
    output_path = os.path.join(video_dir, output_name)
    if not force and os.path.exists(output_path):
        return "skipped"

    audio_path = os.path.join(video_dir, "audio.wav")
    if not os.path.exists(audio_path):
        return f"missing audio.wav"

    # Get total video frame count from metadata in vae_latents.pt or from video file
    total_frames = None
    vae_path = os.path.join(video_dir, "vae_latents.pt")
    if os.path.exists(vae_path):
        try:
            vae_data = torch.load(vae_path, map_location="cpu")
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

    # Load audio
    try:
        audio_np, _ = librosa.load(audio_path, sr=16000)
    except Exception as e:
        return f"audio load error: {e}"

    # Encode
    try:
        audio_emb = encode_audio(wav2vec, feature_extractor, audio_np, total_frames, device)
    except Exception as e:
        return f"encode error: {e}"

    # Save
    torch.save({
        "audio_emb": audio_emb.squeeze(0).cpu(),  # [total_frames, 10752]
        "metadata": {
            "total_video_frames": total_frames,
            "fps": 25,
            "audio_length_samples": len(audio_np),
            "feature_dim": audio_emb.shape[-1],
        },
    }, output_path)

    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Precompute OmniAvatar audio embeddings")
    parser.add_argument("--wav2vec_path", type=str, required=True)
    parser.add_argument("--data_list_path", type=str, required=True,
                        help="Text file with one video directory per line")
    parser.add_argument("--output_name", type=str, default="audio_emb_omniavatar.pt",
                        help="Output filename in each video directory")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    # Load video directories
    with open(args.data_list_path) as f:
        all_dirs = [line.strip() for line in f if line.strip()]

    # Shard
    shard_dirs = all_dirs[args.shard_id::args.num_shards]
    print(f"[Shard {args.shard_id}/{args.num_shards}] Processing {len(shard_dirs)}/{len(all_dirs)} directories")

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
            video_dir, wav2vec, feature_extractor, args.device, args.output_name, args.force,
        )

        if result == "ok":
            stats["ok"] += 1
        elif result == "skipped":
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
