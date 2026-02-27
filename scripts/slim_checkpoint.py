#!/usr/bin/env python3
"""Convert an accelerator checkpoint to the lean format.

Reduces checkpoint size from ~45 GB to ~3.6 GB by extracting only trainable
(LoRA + audio) params from the full model.safetensors.

Usage:
    python scripts/slim_checkpoint.py /path/to/checkpoint-500 --in-place
    python scripts/slim_checkpoint.py /path/to/checkpoint-500  # creates checkpoint-500-lean/
"""

import argparse
import os
import sys
import json
import shutil

import torch

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None


TRAINABLE_PATTERNS = [
    "lora_A", "lora_B",
    "audio_proj",
    "audio_cond_projs",
    "patch_embedding",
]


def identify_trainable_keys(state_dict):
    """Identify trainable parameter keys by pattern matching."""
    return {
        key for key in state_dict
        if any(pat in key for pat in TRAINABLE_PATTERNS)
    }


def slim_checkpoint(ckpt_dir, in_place=False):
    output_dir = ckpt_dir if in_place else ckpt_dir.rstrip("/") + "-lean"

    # --- Load full model state ---
    model_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.isfile(model_path):
        model_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if not os.path.isfile(model_path):
        print(f"ERROR: No model.safetensors or pytorch_model.bin in {ckpt_dir}")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    if model_path.endswith(".safetensors"):
        if load_safetensors is None:
            print("ERROR: safetensors package required. Install with: pip install safetensors")
            sys.exit(1)
        full_sd = load_safetensors(model_path)
    else:
        full_sd = torch.load(model_path, map_location="cpu", weights_only=True)

    total_params = sum(v.numel() for v in full_sd.values())
    print(f"Full model: {len(full_sd)} keys, {total_params / 1e9:.2f}B params")

    # --- Extract trainable params ---
    trainable_keys = identify_trainable_keys(full_sd)
    trainable_sd = {k: v for k, v in full_sd.items() if k in trainable_keys}

    # Strip "pipe.dit." prefix to match export_trainable_state_dict format
    stripped_sd = {}
    for k, v in trainable_sd.items():
        new_k = k[len("pipe.dit."):] if k.startswith("pipe.dit.") else k
        stripped_sd[new_k] = v

    trainable_params = sum(v.numel() for v in stripped_sd.values())
    print(f"Trainable: {len(stripped_sd)} keys, {trainable_params / 1e6:.1f}M params")

    # --- Save ---
    if not in_place:
        os.makedirs(output_dir, exist_ok=True)

    trainable_path = os.path.join(output_dir, "trainable_weights.pt")
    torch.save(stripped_sd, trainable_path)
    print(f"Saved trainable weights to {trainable_path} "
          f"({os.path.getsize(trainable_path) / 1e9:.2f} GB)")

    # --- Copy/rename optimizer, scheduler, RNG, wandb_id ---
    for filename in sorted(os.listdir(ckpt_dir)):
        filepath = os.path.join(ckpt_dir, filename)
        if not os.path.isfile(filepath):
            continue

        if filename == "optimizer.bin":
            dst = os.path.join(output_dir, "optimizer.pt")
            if in_place:
                os.rename(filepath, dst)
            else:
                shutil.copy2(filepath, dst)
            print(f"{'Moved' if in_place else 'Copied'} optimizer.bin → optimizer.pt")

        elif filename == "scheduler.bin":
            dst = os.path.join(output_dir, "scheduler.pt")
            if in_place:
                os.rename(filepath, dst)
            else:
                shutil.copy2(filepath, dst)
            print(f"{'Moved' if in_place else 'Copied'} scheduler.bin → scheduler.pt")

        elif filename.startswith("random_states_") and filename.endswith(".pkl"):
            if not in_place:
                shutil.copy2(filepath, os.path.join(output_dir, filename))
            print(f"{'Kept' if in_place else 'Copied'} {filename}")

        elif filename == "wandb_id.txt":
            if not in_place:
                shutil.copy2(filepath, os.path.join(output_dir, filename))
            print(f"{'Kept' if in_place else 'Copied'} wandb_id.txt")

    # --- Save metadata ---
    step_str = os.path.basename(ckpt_dir.rstrip("/")).split("-")[-1]
    try:
        global_step = int(step_str)
    except ValueError:
        global_step = -1
    metadata = {
        "global_step": global_step,
        "format": "lean_v1",
        "converted_from": "accelerator",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Clean up old large files if in-place ---
    if in_place:
        for old_file in ["model.safetensors", "pytorch_model.bin"]:
            old_path = os.path.join(ckpt_dir, old_file)
            if os.path.isfile(old_path):
                os.remove(old_path)
                print(f"Removed {old_file}")

    # --- Summary ---
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    print(f"\nLean checkpoint at {output_dir}: {total_size / 1e9:.2f} GB total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_dir", help="Path to accelerator checkpoint directory")
    parser.add_argument("--in-place", action="store_true",
                        help="Replace files in-place instead of creating a new directory")
    args = parser.parse_args()
    slim_checkpoint(args.checkpoint_dir, in_place=args.in_place)
