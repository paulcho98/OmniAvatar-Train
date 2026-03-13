# Mask-Weighted MSE Loss for V2V Training

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upweight the MSE diffusion loss in the mouth/lower-face region so the model focuses more on the area it's actually regenerating.

**Architecture:** Port StableAvatar's `mouth_weight` strategy into our existing `train_v2v.py`. StableAvatar computes per-element MSE (no reduction), builds a spatial weight map from the mask (`mouth_weight` inside mouth, 1.0 outside), multiplies, then takes the mean. We already have the LatentSync mask loaded and resized to latent resolution — we just need to apply it to the loss.

**Tech Stack:** PyTorch (`F.mse_loss` with `reduction='none'`), existing `self.latentsync_mask` infrastructure.

---

## Reference: StableAvatar's Approach

From `wan_video_new.py:4262-4290`:

```python
# Per-element MSE (no reduction)
loss_per_element = F.mse_loss(noise_pred.float(), training_target.float(), reduction='none')
# loss_per_element shape: [B, 16, Tzip, H8, W8]

# mask_rgb_t: [B, 1, T, H, W] with 0.0=mouth, 1.0=keep
# Resized to latent space via trilinear interpolation
weight_mask = resize_mask_for_loss_weighting(mask_rgb_t, training_target)

# Invert: mouth_indicator is 1.0 inside mouth, 0.0 outside
mouth_indicator = 1.0 - weight_mask

# weight_map: mouth_weight inside mouth, 1.0 outside
weight_map = mouth_indicator * (mouth_weight - 1.0) + 1.0
# weight_map: [B, 1, Tzip, H8, W8] — broadcasts across 16 channels

loss = (loss_per_element * weight_map).mean()
```

## Our Simplification

StableAvatar's `mask_rgb_t` is at RGB resolution and needs `resize_mask_for_loss_weighting()` (trilinear 3D interpolation) to get to latent space. We don't need that — we already have:

- `self.latentsync_mask`: `[256, 256]`, LatentSync convention (1=keep upper face, 0=mask mouth)
- `self._get_latent_resolution_mask(latents)`: resizes to `[H_lat, W_lat]` via bilinear + threshold, returns `1.0=keep, 0.0=mouth`

So we can build the weight map directly from the existing latent-resolution mask, expanding it to `[1, 1, 1, H_lat, W_lat]` to broadcast across `[B, C, T, H, W]`.

---

## Task 1: Add `--mouth_weight` CLI Arg and Store on Training Module

**Files:**
- Modify: `scripts/train_v2v.py:1988-2007` (argparser, aux losses section)
- Modify: `scripts/train_v2v.py:379-398` (OmniAvatarV2VTrainingModule.__init__)

- [ ] **Step 1: Add CLI argument**

In `train_parser()`, add after the `--aux_num_frames` line (line 1996):

```python
parser.add_argument("--mouth_weight", type=float, default=1.0,
                    help="Weight multiplier for MSE loss in the mouth region. "
                         "1.0 = uniform (disabled). 5.0 = StableAvatar default.")
```

- [ ] **Step 2: Add `mouth_weight` parameter to `__init__`**

Add `mouth_weight=1.0` to the `__init__` signature (after `use_ref_sequence=False`):

```python
def __init__(self, dit_paths, text_encoder_path, vae_path, wav2vec_path,
             omniavatar_ckpt=None, lora_rank=128, lora_alpha=64,
             lora_target_modules="q,k,v,o,ffn.0,ffn.2",
             use_gradient_checkpointing=True,
             use_gradient_checkpointing_offload=False,
             latentsync_mask=None,
             num_training_frames=81,
             dtype=torch.bfloat16,
             mask_all_frames=False,
             no_first_frame_overwrite=False,
             use_ref_sequence=False,
             mouth_weight=1.0):
```

Store it (alongside existing `_last_aux_losses` init pattern):

```python
self.mouth_weight = mouth_weight
self._last_unweighted_mse = None
```

- [ ] **Step 3: Pass `mouth_weight` from main to constructor**

In the `if __name__ == "__main__"` block (~line 2079), add to the `OmniAvatarV2VTrainingModule(...)` call:

```python
mouth_weight=args.mouth_weight,
```

- [ ] **Step 4: Verify no syntax errors**

Run: `python -c "import ast; ast.parse(open('scripts/train_v2v.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/train_v2v.py
git commit -m "feat: add --mouth_weight CLI arg for mask-weighted MSE loss"
```

---

## Task 2: Apply Weighted MSE in Forward Pass

**Files:**
- Modify: `scripts/train_v2v.py:983-986` (flow matching MSE loss section)

- [ ] **Step 1: Replace plain MSE with weighted MSE**

Replace lines 983-986:

```python
# 8. Flow matching MSE loss with timestep weighting
# Apply timestep weight to MSE ONLY (not aux losses) — matching StableAvatar
loss = F.mse_loss(noise_pred.float(), training_target.float())
loss = loss * self.scheduler.training_weight(timestep)
```

With:

```python
# 8. Flow matching MSE loss with spatial mask weighting + timestep weighting
if self.mouth_weight != 1.0 and self.latentsync_mask is not None:
    # Per-element MSE (no reduction) for spatial weighting
    loss_per_element = F.mse_loss(
        noise_pred.float(), training_target.float(), reduction='none'
    )
    # loss_per_element: [1, 16, T_lat, H_lat, W_lat]

    # Build weight map from latent-resolution mask
    # latent_mask: [H_lat, W_lat], 1.0=keep (upper face), 0.0=mouth
    latent_mask = self._get_latent_resolution_mask(noise_pred)
    latent_mask = latent_mask.to(device=device)
    mouth_indicator = 1.0 - latent_mask  # 1.0=mouth, 0.0=keep
    # weight_map: mouth_weight in mouth, 1.0 outside
    weight_map = mouth_indicator * (self.mouth_weight - 1.0) + 1.0
    # Expand to [1, 1, 1, H_lat, W_lat] for broadcasting
    weight_map = weight_map[None, None, None]

    loss = (loss_per_element * weight_map).mean()
else:
    loss = F.mse_loss(noise_pred.float(), training_target.float())
loss = loss * self.scheduler.training_weight(timestep)
```

**Key details:**
- `_get_latent_resolution_mask(noise_pred)` already exists and returns `[H_lat, W_lat]` at the correct resolution. We pass `noise_pred` just for its `.shape` (to read `H_lat`, `W_lat`).
- `weight_map` shape `[1, 1, 1, H_lat, W_lat]` broadcasts cleanly against `loss_per_element` shape `[1, 16, T_lat, H_lat, W_lat]` — same weight applied to all channels and all timesteps.
- When `mouth_weight=1.0` (default), the entire codepath is skipped — identical to current behavior, zero overhead.
- **Frame-0 note:** The weight map applies uniformly across all temporal frames, including frame 0. When `mask_all_frames=False`, frame 0 is the reference frame (mask channel=0, latent overwritten during inference). Applying mouth weighting to frame 0's loss is harmless — the model still reconstructs it faithfully — but if needed later, the weight map could be made `[1, 1, T, H, W]` with frame 0 set to all 1.0s.

- [ ] **Step 2: Verify no syntax errors**

Run: `python -c "import ast; ast.parse(open('scripts/train_v2v.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/train_v2v.py
git commit -m "feat: apply spatial mask weighting to MSE loss in mouth region"
```

---

## Task 3: Log Weighted vs Unweighted MSE for Monitoring

**Files:**
- Modify: `scripts/train_v2v.py` — forward() method, near the MSE loss computation
- Modify: `scripts/train_v2v.py` — wandb logging section

- [ ] **Step 1: Store unweighted MSE alongside weighted MSE**

After the new MSE loss block (from Task 2), add tracking for the unweighted loss so we can compare:

```python
# Store unweighted MSE for monitoring (only when weighting is active)
if self.mouth_weight != 1.0 and self.latentsync_mask is not None:
    self._last_unweighted_mse = loss_per_element.mean().detach()
else:
    self._last_unweighted_mse = None
```

- [ ] **Step 2: Find the wandb logging section and add the metric**

Search for where `mse` is logged to wandb (likely in the training loop). Add:

```python
if hasattr(model_unwrapped, '_last_unweighted_mse') and model_unwrapped._last_unweighted_mse is not None:
    log_dict["mse_unweighted"] = model_unwrapped._last_unweighted_mse.item()
    model_unwrapped._last_unweighted_mse = None
```

- [ ] **Step 3: Log `mouth_weight` in wandb config**

Find where `wandb.init(...)` is called (or where config is passed to it) and add `mouth_weight` to the config dict so it shows up in wandb run comparison:

```python
"mouth_weight": args.mouth_weight,
```

- [ ] **Step 4: Verify no syntax errors**

Run: `python -c "import ast; ast.parse(open('scripts/train_v2v.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/train_v2v.py
git commit -m "feat: log unweighted MSE alongside weighted MSE for comparison"
```

---

## Task 4: Update Launch Scripts

**Files:**
- Modify: whichever `scripts/train_v2v_*.sh` scripts are actively used

- [ ] **Step 1: Identify active launch scripts**

Check which `.sh` scripts are currently used for training (the ones referenced in CLAUDE.md or recently modified). Add `--mouth_weight 5.0` to the ones you want to enable it for. Example:

```bash
# Add to the accelerate launch command args:
    --mouth_weight 5.0 \
```

**Note:** Default is 1.0 (disabled), so existing scripts that don't specify it will behave identically to before.

- [ ] **Step 2: Commit**

```bash
git add scripts/train_v2v_*.sh
git commit -m "feat: enable mouth_weight=5.0 in active training launch scripts"
```

---

## Summary of Changes

| File | Change | Lines |
|---|---|---|
| `scripts/train_v2v.py` | Add `--mouth_weight` CLI arg | ~1 line in argparser |
| `scripts/train_v2v.py` | Add `mouth_weight` to `__init__` + store | ~3 lines |
| `scripts/train_v2v.py` | Pass `mouth_weight` from main | ~1 line |
| `scripts/train_v2v.py` | Weighted MSE in `forward()` | ~15 lines replacing 3 |
| `scripts/train_v2v.py` | Log unweighted MSE | ~5 lines |
| `scripts/train_v2v_*.sh` | Add `--mouth_weight 5.0` | ~1 line per script |

**Total: ~25 lines of new code.** No new files, no new dependencies, no new functions.
