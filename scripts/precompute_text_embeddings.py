"""Precompute T5 (UMT5-XXL) text embeddings for V2V training data.

Reads a metadata CSV, encodes each unique prompt once through the T5 encoder,
saves each as a .pt file named by the MD5 hash of the prompt text, and
creates text_emb.pt symlinks in each sample directory pointing to the correct
pre-computed embedding file.

Output tensors are [1, 512, 4096] fp32 to match existing hallo3_data embeddings
(the dataloader casts to bf16 at load time).

Usage:
    python scripts/precompute_text_embeddings.py \\
        --csv_path /path/to/metadata.csv \\
        --text_encoder_path pretrained_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth \\
        --output_dir /path/to/text_emb/ \\
        --sample_dir /path/to/training_data/

Sharded execution (e.g. 2 shards for 2 GPUs):
    CUDA_VISIBLE_DEVICES=0 python scripts/precompute_text_embeddings.py \\
        --csv_path ... --text_encoder_path ... --output_dir ... --sample_dir ... \\
        --shard_id 0 --num_shards 2 --device cuda:0 &
    CUDA_VISIBLE_DEVICES=1 python scripts/precompute_text_embeddings.py \\
        --csv_path ... --text_encoder_path ... --output_dir ... --sample_dir ... \\
        --shard_id 1 --num_shards 2 --device cuda:0 &
    wait

Note: --symlink_only skips encoding and only (re)creates the symlinks using
already-saved .pt files in output_dir.
"""

import argparse
import hashlib
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm

# Add project root to sys.path so OmniAvatar can be imported directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.wan_video import WanVideoPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md5_of_string(text: str) -> str:
    """Return the 32-character hex MD5 digest of *text* (UTF-8 encoded)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_pipeline(text_encoder_path: str, device: str) -> WanVideoPipeline:
    """Load T5 text encoder via ModelManager and return a WanVideoPipeline.

    Only the text encoder (and its tokenizer) is loaded — no DiT, no VAE.
    The model is placed on CPU first and moved to *device* just before
    encoding (mirrors the training script pattern).
    """
    print(f"[load] Loading T5 text encoder from: {text_encoder_path}")
    model_manager = ModelManager(device="cpu", infer=True)
    model_manager.load_model(text_encoder_path, torch_dtype=torch.bfloat16, device="cpu")

    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    pipe.text_encoder.eval()
    pipe.text_encoder.to(device)
    print(f"[load] Text encoder ready on {device}")
    return pipe


@torch.no_grad()
def encode_prompt(pipe: WanVideoPipeline, prompt: str, device: str) -> torch.Tensor:
    """Encode a single prompt string → fp32 tensor of shape [1, 512, 4096].

    Uses the same encode_prompt path as the training script
    (train_v2v.py:647):  prompter.encode_prompt(prompt, positive=True, device=device)

    The prompter zeroes out padding positions beyond the actual token length,
    so the result is directly usable without further processing.

    The output is cast to fp32 to match existing hallo3_data embeddings.
    """
    emb = pipe.prompter.encode_prompt(prompt, positive=True, device=device)
    # emb shape: [1, 512, 4096] (batch=1, text_len=512, dim=4096)
    return emb.to(dtype=torch.float32).cpu()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def encode_unique_prompts(
    unique_prompts: list,
    pipe: WanVideoPipeline,
    output_dir: str,
    device: str,
    batch_size: int = 1,
) -> dict:
    """Encode every prompt in *unique_prompts* that does not already have a
    saved .pt file in *output_dir*.

    Returns a mapping  prompt_text -> absolute_path_to_pt_file.
    """
    os.makedirs(output_dir, exist_ok=True)

    prompt_to_path: dict = {}
    to_encode: list = []

    for prompt in unique_prompts:
        key = md5_of_string(prompt)
        out_path = os.path.join(output_dir, f"{key}.pt")
        prompt_to_path[prompt] = out_path
        if not os.path.exists(out_path):
            to_encode.append(prompt)
        # else: already computed, nothing to do

    if not to_encode:
        print("[encode] All embeddings already exist — nothing to encode.")
        return prompt_to_path

    print(f"[encode] Encoding {len(to_encode)} new prompts "
          f"(skipping {len(unique_prompts) - len(to_encode)} cached)")

    # Process in batches so tqdm progress is meaningful even at batch_size=1.
    for i in tqdm(range(0, len(to_encode), batch_size), desc="Encoding prompts"):
        batch = to_encode[i : i + batch_size]
        for prompt in batch:
            out_path = prompt_to_path[prompt]
            emb = encode_prompt(pipe, prompt, device)  # [1, 512, 4096] fp32
            torch.save(emb, out_path)

    print(f"[encode] Done. Saved to {output_dir}")
    return prompt_to_path


def create_symlinks(
    df: pd.DataFrame,
    prompt_to_path: dict,
    sample_dir: str,
    overwrite: bool = False,
) -> None:
    """Create (or refresh) text_emb.pt symlinks in each sample directory.

    For each row in *df* the sample directory is identified as
    ``<sample_dir>/<video_stem>`` where *video_stem* is the last component
    of the ``video`` column path (without extension).

    The symlink target is the absolute path to the pre-computed .pt file.

    Parameters
    ----------
    df:
        Full metadata dataframe with at least ``video`` and ``prompt`` columns.
    prompt_to_path:
        Mapping from prompt text to the absolute path of its .pt file.
    sample_dir:
        Root directory containing per-sample sub-directories.
    overwrite:
        If True, replace existing symlinks/files. If False, skip existing ones.
    """
    missing_dirs = 0
    created = 0
    skipped = 0
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating symlinks"):
        video_col = str(row["video"])
        prompt = str(row["prompt"])

        # Derive sample directory name from the video path's basename (no ext).
        video_stem = os.path.splitext(os.path.basename(video_col))[0]
        dir_path = os.path.join(sample_dir, video_stem)

        if not os.path.isdir(dir_path):
            # Try using the video column value directly as a sub-path.
            alt = os.path.join(sample_dir, video_col)
            if os.path.isdir(alt):
                dir_path = alt
            else:
                missing_dirs += 1
                if missing_dirs <= 5:
                    print(f"[symlink] WARNING: directory not found for video={video_col!r} "
                          f"(tried {dir_path} and {alt})")
                continue

        target = prompt_to_path.get(prompt)
        if target is None:
            print(f"[symlink] ERROR: no embedding found for prompt: {prompt[:80]!r}")
            errors += 1
            continue

        link_path = os.path.join(dir_path, "text_emb.pt")

        if os.path.lexists(link_path):
            if not overwrite:
                skipped += 1
                continue
            os.remove(link_path)

        try:
            os.symlink(os.path.abspath(target), link_path)
            created += 1
        except OSError as exc:
            print(f"[symlink] ERROR creating symlink at {link_path}: {exc}")
            errors += 1

    if missing_dirs > 5:
        print(f"[symlink] ... and {missing_dirs - 5} more missing directories (total {missing_dirs})")
    print(f"[symlink] Created={created}  Skipped={skipped}  "
          f"MissingDirs={missing_dirs}  Errors={errors}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute T5 text embeddings and create per-sample symlinks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path to metadata CSV with columns: video, prompt, text_emb, audio_emb, masks, total_frames",
    )
    parser.add_argument(
        "--text_encoder_path", type=str,
        default="pretrained_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path to the UMT5-XXL .pth file.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where per-prompt <md5>.pt files are saved.",
    )
    parser.add_argument(
        "--sample_dir", type=str, default=None,
        help=(
            "Root directory containing per-sample sub-directories. "
            "If provided, a text_emb.pt symlink is created in each sub-directory "
            "pointing to the appropriate embedding in output_dir."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device to run T5 encoding on.",
    )
    parser.add_argument(
        "--overwrite_symlinks", action="store_true",
        help="Replace existing text_emb.pt symlinks (default: skip).",
    )
    parser.add_argument(
        "--symlink_only", action="store_true",
        help=(
            "Skip encoding entirely. Only create/refresh symlinks using "
            "the .pt files already present in output_dir."
        ),
    )
    # Sharding support for parallel execution across multiple GPUs.
    parser.add_argument(
        "--shard_id", type=int, default=0,
        help="Index of this shard (0-based). Used with --num_shards.",
    )
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help="Total number of shards. Prompts are split deterministically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Read CSV and extract unique prompts
    # ------------------------------------------------------------------
    print(f"[main] Reading CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    required_cols = {"video", "prompt"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["video"] = df["video"].fillna("").astype(str)

    all_unique = sorted(df["prompt"].unique().tolist())
    print(f"[main] Total rows: {len(df)}  |  Unique prompts: {len(all_unique)}")

    # ------------------------------------------------------------------
    # 2. Apply sharding (deterministic: sorted order then stride slice)
    # ------------------------------------------------------------------
    if args.num_shards > 1:
        shard_prompts = all_unique[args.shard_id :: args.num_shards]
        print(f"[main] Shard {args.shard_id}/{args.num_shards}: "
              f"{len(shard_prompts)} prompts to encode")
    else:
        shard_prompts = all_unique

    # ------------------------------------------------------------------
    # 3. Encode unique prompts (unless --symlink_only)
    # ------------------------------------------------------------------
    if not args.symlink_only:
        pipe = load_pipeline(args.text_encoder_path, args.device)
        prompt_to_path = encode_unique_prompts(
            shard_prompts, pipe, args.output_dir, args.device
        )
        # Release GPU memory after encoding.
        pipe.text_encoder.to("cpu")
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("[main] --symlink_only: skipping encoding")
        # Build prompt_to_path from what is already on disk.
        prompt_to_path = {}
        for prompt in all_unique:
            key = md5_of_string(prompt)
            out_path = os.path.join(args.output_dir, f"{key}.pt")
            if os.path.exists(out_path):
                prompt_to_path[prompt] = out_path
            # Prompts without a saved file are silently skipped;
            # the symlink step will report them as errors.

    # ------------------------------------------------------------------
    # 4. Create symlinks in sample directories
    # ------------------------------------------------------------------
    if args.sample_dir is not None:
        # When sharding, we still create symlinks for ALL rows whose prompt
        # now has a saved embedding (including those from previous shards).
        # Refresh prompt_to_path to cover the full prompt set.
        full_prompt_to_path: dict = {}
        for prompt in all_unique:
            key = md5_of_string(prompt)
            out_path = os.path.join(args.output_dir, f"{key}.pt")
            if os.path.exists(out_path):
                full_prompt_to_path[prompt] = out_path

        create_symlinks(
            df,
            full_prompt_to_path,
            args.sample_dir,
            overwrite=args.overwrite_symlinks,
        )
    else:
        print("[main] --sample_dir not set; skipping symlink creation.")

    print("[main] All done.")


if __name__ == "__main__":
    main()
