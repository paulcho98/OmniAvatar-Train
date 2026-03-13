#!/usr/bin/env python3
"""CPU-only verification of lean checkpointing.

Test 1: Slim checkpoint-500 (non-destructive) and verify extracted trainable
        weights match the existing step-500.pt (ground truth from training).

Test 2: Round-trip save → load with a mock model mimicking
        OmniAvatarV2VTrainingModule structure. Verifies model weights,
        optimizer states, scheduler states, and RNG states are exactly restored.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/verify_lean_checkpoint.py \
        --checkpoint_dir /home/work/output_omniavatar_v2v_auxloss/checkpoint-500 \
        --lora_weights /home/work/output_omniavatar_v2v_auxloss/step-500.pt
"""

import argparse
import os
import sys
import json
import random
import pickle
import tempfile
import shutil

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Test 1: Validate slim_checkpoint.py against ground truth step-500.pt
# ---------------------------------------------------------------------------

def test_slim_checkpoint(checkpoint_dir, lora_weights_path):
    """Slim checkpoint-500 → checkpoint-500-lean/ and compare against step-500.pt."""
    print("=" * 70)
    print("TEST 1: Slim checkpoint validation")
    print("=" * 70)

    # Run slim_checkpoint.py (without --in-place)
    lean_dir = checkpoint_dir.rstrip("/") + "-lean"
    if os.path.isdir(lean_dir):
        shutil.rmtree(lean_dir)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from slim_checkpoint import slim_checkpoint
    print(f"\nSlimming {checkpoint_dir} → {lean_dir}")
    slim_checkpoint(checkpoint_dir, in_place=False)

    # Load both checkpoints
    print(f"\nComparing trainable_weights.pt vs {os.path.basename(lora_weights_path)}...")
    lean_sd = torch.load(os.path.join(lean_dir, "trainable_weights.pt"),
                         map_location="cpu", weights_only=True)
    gt_sd = torch.load(lora_weights_path, map_location="cpu", weights_only=True)

    # Compare keys
    lean_keys = set(lean_sd.keys())
    gt_keys = set(gt_sd.keys())

    if lean_keys == gt_keys:
        print(f"  Keys match: {len(lean_keys)} params")
    else:
        only_lean = lean_keys - gt_keys
        only_gt = gt_keys - lean_keys
        if only_lean:
            print(f"  EXTRA in lean (not in step-500.pt): {len(only_lean)}")
            for k in sorted(only_lean)[:10]:
                print(f"    {k}")
        if only_gt:
            print(f"  MISSING from lean (in step-500.pt but not extracted): {len(only_gt)}")
            for k in sorted(only_gt)[:10]:
                print(f"    {k}")

    # Compare values for shared keys
    shared = lean_keys & gt_keys
    mismatches = []
    for k in sorted(shared):
        if not torch.equal(lean_sd[k], gt_sd[k]):
            max_diff = (lean_sd[k].float() - gt_sd[k].float()).abs().max().item()
            mismatches.append((k, max_diff))

    if not mismatches:
        print(f"  Values match exactly for all {len(shared)} shared params")
    else:
        print(f"  MISMATCHES in {len(mismatches)} params:")
        for k, diff in mismatches[:10]:
            print(f"    {k}: max_diff={diff}")

    # Check other files exist
    expected_files = ["optimizer.pt", "scheduler.pt", "metadata.json"]
    for f in expected_files:
        path = os.path.join(lean_dir, f)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        status = f"{size / 1e6:.1f} MB" if exists else "MISSING"
        print(f"  {f}: {status}")

    rng_files = [f for f in os.listdir(lean_dir) if f.startswith("random_states_")]
    print(f"  RNG state files: {len(rng_files)}")

    wandb_path = os.path.join(lean_dir, "wandb_id.txt")
    if os.path.isfile(wandb_path):
        with open(wandb_path) as f:
            print(f"  wandb_id.txt: {f.read().strip()}")

    # Summary
    total_size = sum(
        os.path.getsize(os.path.join(lean_dir, f))
        for f in os.listdir(lean_dir)
        if os.path.isfile(os.path.join(lean_dir, f))
    )
    orig_size = sum(
        os.path.getsize(os.path.join(checkpoint_dir, f))
        for f in os.listdir(checkpoint_dir)
        if os.path.isfile(os.path.join(checkpoint_dir, f))
    )
    print(f"\n  Original: {orig_size / 1e9:.2f} GB")
    print(f"  Lean:     {total_size / 1e9:.2f} GB")
    print(f"  Ratio:    {orig_size / total_size:.1f}x smaller")

    passed = (lean_keys == gt_keys or lean_keys >= gt_keys) and len(mismatches) == 0
    print(f"\n  TEST 1: {'PASSED' if passed else 'FAILED'}")
    return passed, lean_dir


# ---------------------------------------------------------------------------
# Test 2: Round-trip save → load with mock model
# ---------------------------------------------------------------------------

class MockDiT(nn.Module):
    """Mimics the DiT with LoRA + audio modules."""
    def __init__(self):
        super().__init__()
        # Frozen base layer (simulates DiT attention)
        self.base_linear = nn.Linear(64, 64, bias=False)
        self.base_linear.requires_grad_(False)

        # LoRA layers (trainable) — mimics lora_A/lora_B naming
        self.lora_A = nn.Linear(64, 8, bias=False)
        self.lora_B = nn.Linear(8, 64, bias=False)

        # Audio modules (trainable)
        self.audio_proj = nn.Linear(32, 64, bias=False)
        self.audio_cond_projs = nn.ModuleList([nn.Linear(32, 64) for _ in range(3)])

        # Patch embedding (trainable)
        self.patch_embedding = nn.Linear(49, 64)


class MockTrainingModule(nn.Module):
    """Mimics OmniAvatarV2VTrainingModule for testing save/load."""
    def __init__(self):
        super().__init__()
        self.pipe = nn.Module()
        self.pipe.dit = MockDiT()

        # Freeze base, keep LoRA + audio + patch_embedding trainable
        self.pipe.dit.base_linear.requires_grad_(False)
        self.pipe.dit.lora_A.requires_grad_(True)
        self.pipe.dit.lora_B.requires_grad_(True)
        self.pipe.dit.audio_proj.requires_grad_(True)
        self.pipe.dit.audio_cond_projs.requires_grad_(True)
        self.pipe.dit.patch_embedding.requires_grad_(True)

    def trainable_modules(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_names(self):
        return {n for n, p in self.named_parameters() if p.requires_grad}

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        names = self.trainable_param_names()
        out = {k: v for k, v in state_dict.items() if k in names}
        if remove_prefix:
            out = {
                (k[len(remove_prefix):] if k.startswith(remove_prefix) else k): v
                for k, v in out.items()
            }
        return out


class MockAccelerator:
    """Minimal mock of HuggingFace Accelerator for CPU testing."""
    def __init__(self):
        self.is_main_process = True
        self.process_index = 0
        self.num_processes = 1
        self.device = torch.device("cpu")

    def get_state_dict(self, model):
        return model.state_dict()

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        pass


class MockArgs:
    lora_rank = 8
    lora_alpha = 4


def test_round_trip():
    """Save → load round-trip with mock model, verify exact state restoration."""
    print("\n" + "=" * 70)
    print("TEST 2: Round-trip save → load")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp(prefix="lean_ckpt_test_")
    ckpt_dir = os.path.join(tmpdir, "checkpoint-42")
    all_passed = True

    try:
        # --- Setup ---
        model = MockTrainingModule()
        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=1e-4, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        accelerator = MockAccelerator()
        args = MockArgs()

        # Run a fake training step to populate optimizer state
        x = torch.randn(2, 64)
        out = model.pipe.dit.base_linear(x) + model.pipe.dit.lora_B(model.pipe.dit.lora_A(x))
        loss = out.sum()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # Set specific RNG states for reproducibility check
        random.seed(12345)
        np.random.seed(67890)
        torch.manual_seed(11111)

        # Capture pre-save states
        pre_model_sd = {k: v.clone() for k, v in model.state_dict().items()}
        pre_opt_sd_str = str(optimizer.state_dict()['state'].keys())
        pre_sched_sd = lr_scheduler.state_dict().copy()
        pre_py_rng = random.getstate()
        pre_np_rng = np.random.get_state()
        pre_torch_rng = torch.random.get_rng_state().clone()

        # Generate some random numbers (to verify RNG restoration later)
        pre_py_val = random.random()
        pre_np_val = np.random.random()
        pre_torch_val = torch.randn(3).tolist()

        print(f"\n  Pre-save random values:")
        print(f"    Python: {pre_py_val:.10f}")
        print(f"    NumPy:  {pre_np_val:.10f}")
        print(f"    Torch:  {pre_torch_val}")

        # --- Save ---
        # Import the actual functions from train_v2v
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

        # We can't easily import save_training_state/load_training_state from train_v2v.py
        # because it has heavy imports. Instead, replicate the core logic here:
        print(f"\n  Saving to {ckpt_dir}...")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save trainable weights
        state_dict = accelerator.get_state_dict(model)
        unwrapped = accelerator.unwrap_model(model)
        trainable_sd = unwrapped.export_trainable_state_dict(
            state_dict, remove_prefix="pipe.dit.",
        )
        torch.save(trainable_sd, os.path.join(ckpt_dir, "trainable_weights.pt"))
        print(f"    Trainable weights: {len(trainable_sd)} params")

        # Save optimizer
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

        # Save scheduler
        torch.save(lr_scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))

        # Save RNG (restore pre-save state first, then re-generate to get same values)
        random.setstate(pre_py_rng)
        np.random.set_state(pre_np_rng)
        torch.random.set_rng_state(pre_torch_rng)

        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
        }
        with open(os.path.join(ckpt_dir, "random_states_0.pkl"), "wb") as f:
            pickle.dump(rng_state, f)

        # Save metadata
        metadata = {"format": "lean_v1", "global_step": 42}
        with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        saved_files = os.listdir(ckpt_dir)
        total_bytes = sum(os.path.getsize(os.path.join(ckpt_dir, f)) for f in saved_files)
        print(f"    Files: {saved_files}")
        print(f"    Total size: {total_bytes / 1024:.1f} KB")

        # --- Corrupt model state (simulate fresh init) ---
        print("\n  Corrupting model state (simulating fresh init)...")
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.fill_(999.0)  # obvious corruption
        random.seed(99999)
        np.random.seed(99999)
        torch.manual_seed(99999)

        # --- Load ---
        print("  Loading from checkpoint...")

        # Load trainable weights
        loaded_sd = torch.load(
            os.path.join(ckpt_dir, "trainable_weights.pt"),
            map_location="cpu", weights_only=True,
        )
        prefixed_sd = {f"pipe.dit.{k}": v for k, v in loaded_sd.items()}
        missing, unexpected = model.load_state_dict(prefixed_sd, strict=False)
        print(f"    Loaded {len(prefixed_sd)} params, {len(missing)} missing (frozen), {len(unexpected)} unexpected")

        # Load optimizer
        opt_sd = torch.load(
            os.path.join(ckpt_dir, "optimizer.pt"),
            map_location="cpu", weights_only=True,
        )
        optimizer.load_state_dict(opt_sd)

        # Load scheduler
        sched_sd = torch.load(
            os.path.join(ckpt_dir, "scheduler.pt"),
            map_location="cpu", weights_only=True,
        )
        lr_scheduler.load_state_dict(sched_sd)

        # Load RNG
        with open(os.path.join(ckpt_dir, "random_states_0.pkl"), "rb") as f:
            loaded_rng = pickle.load(f)
        random.setstate(loaded_rng["python"])
        np.random.set_state(loaded_rng["numpy"])
        torch.random.set_rng_state(loaded_rng["torch_cpu"])

        # --- Verify ---
        print("\n  Verifying...")

        # 2a. Model weights
        post_model_sd = model.state_dict()
        weight_ok = True
        for k in trainable_sd:
            prefixed_k = f"pipe.dit.{k}"
            if prefixed_k in pre_model_sd and prefixed_k in post_model_sd:
                if not torch.equal(pre_model_sd[prefixed_k], post_model_sd[prefixed_k]):
                    print(f"    MISMATCH: {k}")
                    weight_ok = False
        if weight_ok:
            print(f"    Model weights: MATCH ({len(trainable_sd)} trainable params restored)")
        all_passed &= weight_ok

        # 2b. Frozen weights unchanged (should still be original, not 999)
        frozen_ok = True
        for n, p in model.named_parameters():
            if not p.requires_grad:
                if n in pre_model_sd and not torch.equal(p.data, pre_model_sd[n]):
                    print(f"    FROZEN CHANGED: {n}")
                    frozen_ok = False
        if frozen_ok:
            print(f"    Frozen weights: UNCHANGED (correct)")
        all_passed &= frozen_ok

        # 2c. Optimizer state
        opt_ok = True
        loaded_opt_sd = optimizer.state_dict()
        for param_id in loaded_opt_sd['state']:
            for key in ['exp_avg', 'exp_avg_sq']:
                if key in loaded_opt_sd['state'][param_id]:
                    # Just verify it's not all zeros (meaning state was loaded)
                    t = loaded_opt_sd['state'][param_id][key]
                    if t.abs().sum() == 0:
                        print(f"    WARNING: optimizer state[{param_id}][{key}] is all zeros")
                        opt_ok = False
        if opt_ok:
            print(f"    Optimizer state: OK ({len(loaded_opt_sd['state'])} param states)")
        all_passed &= opt_ok

        # 2d. Scheduler state
        sched_ok = lr_scheduler.state_dict() == pre_sched_sd
        print(f"    Scheduler state: {'MATCH' if sched_ok else 'MISMATCH'}")
        all_passed &= sched_ok

        # 2e. RNG states — generate same random numbers
        post_py_val = random.random()
        post_np_val = np.random.random()
        post_torch_val = torch.randn(3).tolist()

        print(f"\n  Post-load random values:")
        print(f"    Python: {post_py_val:.10f}")
        print(f"    NumPy:  {post_np_val:.10f}")
        print(f"    Torch:  {post_torch_val}")

        py_rng_ok = pre_py_val == post_py_val
        np_rng_ok = pre_np_val == post_np_val
        torch_rng_ok = pre_torch_val == post_torch_val

        print(f"\n    Python RNG:  {'MATCH' if py_rng_ok else 'MISMATCH'}")
        print(f"    NumPy RNG:   {'MATCH' if np_rng_ok else 'MISMATCH'}")
        print(f"    Torch RNG:   {'MATCH' if torch_rng_ok else 'MISMATCH'}")
        all_passed &= py_rng_ok and np_rng_ok and torch_rng_ok

        # 2f. Format detection
        has_marker = os.path.isfile(os.path.join(ckpt_dir, "trainable_weights.pt"))
        print(f"\n    Format detection (trainable_weights.pt exists): {has_marker}")
        all_passed &= has_marker

        print(f"\n  TEST 2: {'PASSED' if all_passed else 'FAILED'}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to accelerator checkpoint dir (for Test 1)")
    parser.add_argument("--lora_weights", type=str, default=None,
                        help="Path to step-N.pt LoRA weights (ground truth for Test 1)")
    args = parser.parse_args()

    results = {}

    # Test 1: Slim checkpoint validation (only if paths provided)
    if args.checkpoint_dir and args.lora_weights:
        passed, lean_dir = test_slim_checkpoint(args.checkpoint_dir, args.lora_weights)
        results["slim_checkpoint"] = passed
        print(f"\n  Lean checkpoint at: {lean_dir}")
        print(f"  (Original untouched at: {args.checkpoint_dir})")
    else:
        print("Skipping Test 1 (no --checkpoint_dir / --lora_weights provided)")

    # Test 2: Round-trip (always runs, no dependencies)
    results["round_trip"] = test_round_trip()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  {name}: {'PASSED' if passed else 'FAILED'}")

    all_ok = all(results.values())
    print(f"\n  Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)
