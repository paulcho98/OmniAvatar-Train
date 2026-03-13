# V2V Adaptation Feasibility: Adding Masked Video to OmniAvatar

## Context

OmniAvatar is an audio-driven portrait animation model built on Wan 2.1. It currently
uses a T2V-based approach where the DiT receives: noisy latent (16ch) + reference frame
repeated across time (16ch) + binary temporal mask (1ch) = 33 channels total.

StableAvatar's V2V lip sync training takes a different approach: it feeds the model a
**masked video** (video with the mouth region zeroed out) as explicit conditioning. This
gives the model the upper face as "given" information, reducing the task to mouth-only
reconstruction. See `docs/input-comparison-omniavatar-vs-stableavatar.md` for the full
structural comparison.

**This document records our analysis of adapting OmniAvatar's training to incorporate
the V2V masked-video concept** — specifically:
1. Whether the input modification is architecturally feasible
2. Whether StableAvatar's precomputed VAE latents can be directly reused
3. How the VAE handles single-frame reference encoding (temporal isolation guarantee)

---

## Proposed Input Modification

Keep the existing noise and reference frame inputs. Modify the mask from binary temporal
to spatial face mask. Add masked video latents as a new 16-channel input.

**New 49-channel layout:**
```
x  (Channels 0-15):  noisy latent              (unchanged)
y  (Channels 16-31): reference frame repeated   (unchanged — first frame VAE latent × T_lat)
y  (Channel 32):     spatial face mask          (MODIFIED — was binary temporal, now LatentSync spatial)
y  (Channels 33-48): masked video latents       (NEW — VAE encoding of mouth-erased video)
```

In code, `y` goes from `(B, 17, T_lat, 64, 64)` to `(B, 33, T_lat, 64, 64)`.
Total patch_embedding input: 16 + 33 = 49 channels (was 16 + 17 = 33).

---

## Finding 1: The Input Modification is Architecturally Feasible

### Patch embedding has no temporal compression

The OmniAvatar 14B DiT uses `patch_size = (1, 2, 2)` with `stride = (1, 2, 2)`.

**Source:** `OmniAvatar/models/wan_video_dit.py:281-282`
```python
self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
# For 14B: Conv3d(33, 5120, kernel_size=(1,2,2), stride=(1,2,2))
```

Temporal stride = 1 means each latent frame's spatial content is processed independently.
A spatial mask that is constant across time (except frame 0 = all zeros) works perfectly.

### The mask is a latent-level tensor, never VAE-encoded

The mask is concatenated at the channel dimension in latent space. It just needs to match
the latent spatial resolution: `(B, 1, T_lat, 64, 64)` for 512px inputs (8x spatial
downsampling by VAE). Since the LatentSync mask is the same spatial pattern at every frame:
- Frame 0: all 0s (entire reference frame is "known")
- Frames 1+: LatentSync mask values (0=keep upper face, 1=generate mouth)

StableAvatar uses 4 channels for the mask because it packs 4 per-original-frame mask values
per latent timestep (to handle the VAE's 4x temporal compression). But with a fixed spatial
mask, all 4 frames within a temporal group have the same mask — making the 4-channel grouping
redundant. A single channel suffices.

### smart_load_weights handles the 33→49ch expansion

**Source:** `OmniAvatar/utils/io_utils.py:102-125`

When a model parameter is larger than the checkpoint parameter in any dimension,
`smart_load_weights()` copies the checkpoint weights into the leading portion and leaves
the rest at their initialization. For training mode, new parameters get
`xavier_uniform_(gain=0.05)` initialization before weight loading.

So expanding `in_dim` from 33 to 49 in the model config automatically:
- Copies the existing 33ch `patch_embedding` weights into channels 0-32
- Initializes channels 33-48 (masked video) near zero
- No manual weight surgery needed

---

## Finding 2: StableAvatar's Precomputed VAE Latents Are Directly Compatible

### Same VAE, same normalization, same dtype

| Property | StableAvatar | OmniAvatar | Compatible? |
|----------|-------------|-----------|-------------|
| VAE architecture | AutoencoderKLWan (Wan 2.1) | WanVideoVAE (Wan 2.1) | Identical |
| Frame normalization | `x / 127.5 - 1.0` | `x * 2.0/255.0 - 1.0` | Mathematically identical |
| Output dtype | bf16 | bf16 | Yes |
| Latent channels | 16 | 16 | Yes |
| Mean/std constants | Hardcoded Wan 2.1 stats | Same hardcoded stats | Identical |

Both `x / 127.5 - 1.0` and `x * 2.0/255.0 - 1.0` map `[0, 255] → [-1, 1]` identically
because `1/127.5 = 2/255`.

### Which precomputed files are usable

**`vae_latents.pt` (per video directory):**
- `input_latents [16, 21, 64, 64]` — Full ground truth video, VAE-encoded.
  Use for: noise target computation, loss target, AND extracting ref frame via `[:, :1]`.
- `masked_latents [16, 21, 64, 64]` — Video with mouth zeroed out, VAE-encoded.
  Frame 0 is untouched (identical to `input_latents[:, :1]`).
  Use for: the new 16ch masked video conditioning channels.
- `reference_frame [3, H, W]` — Raw pixel reference frame.
  Use for: visualization only (OmniAvatar doesn't use CLIP).

**`ref_latents.pt` (per video directory):**
- `ref_sequence_latents [16, 21, 64, 64]` — **NOT usable.** This is a VAE encoding
  of a separate 81-frame segment (non-overlapping with the training clip). OmniAvatar's
  reference is the first frame of the training clip repeated, not a separate segment.

### Loading code for the proposed setup

```python
vae_data = torch.load(os.path.join(video_dir, "vae_latents.pt"), map_location="cpu")
input_latents = vae_data['input_latents'].unsqueeze(0)    # (1, 16, 21, 64, 64) — GT
masked_latents = vae_data['masked_latents'].unsqueeze(0)   # (1, 16, 21, 64, 64) — masked
ref_latent = input_latents[:, :, :1]                        # (1, 16, 1, 64, 64) — frame 0

# Build y tensor:
ref_repeated = ref_latent.repeat(1, 1, T_lat, 1, 1)        # (1, 16, 21, 64, 64)
mask = build_spatial_mask(latentsync_mask, T_lat)            # (1, 1, 21, 64, 64)
y = torch.cat([ref_repeated, mask, masked_latents], dim=1)  # (1, 33, 21, 64, 64)
```

---

## Finding 3: The VAE's First Latent Frame is a Clean Single-Frame Encoding

### The Wan 2.1 VAE uses causal chunk-based encoding

**Source:** `OmniAvatar/models/wan_video_vae.py:548-573`
```python
def encode(self, x, scale):
    self.clear_cache()                  # Reset ALL feature caches to [None] * N
    t = x.shape[2]
    iter_ = 1 + (t - 1) // 4           # 81 frames → 21 iterations

    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(x[:, :, :1, :, :], ...)    # ONLY frame 0
        else:
            out_ = self.encoder(x[:, :, 1+4*(i-1):1+4*i, :, :], ...)
            out = torch.cat([out, out_], 2)
```

### Three guarantees that frame 0's latent is temporally pure

1. **Iteration 0 processes frame 0 alone**: `x[:, :, :1]` — a single frame.
   No other frames are fed to the encoder during this iteration.

2. **Causal convolutions cannot look forward**: `CausalConv3d` (`wan_video_vae.py:33-52`)
   applies padding `(2*p, 0)` — only on the past side. With 1 frame and no cache,
   there is no future information to leak.

3. **Cache is freshly cleared**: `clear_cache()` sets `_enc_feat_map = [None] * N`
   (`wan_video_vae.py:614-621`). Iteration 0 sees all-None caches — zero temporal context.

### Conclusion

`vae.encode(full_81_frame_video)[:, :, :1]` is **bitwise identical** to
`vae.encode(single_frame_0)`. The temporal compression (two stride-2 causal conv layers
= 4x total) does not contaminate frame 0 because it's processed in isolation before
any other frames, with zero-initialized caches and causal-only padding.

This means `input_latents[:, :1]` from StableAvatar's precomputed data is a clean
reference frame latent that can be directly repeated across the temporal dimension,
matching exactly what OmniAvatar's `prepare_reference_input()` expects.

---

## Key File References

- `OmniAvatar/models/wan_video_dit.py:281-282` — patch_embedding Conv3d (kernel=(1,2,2))
- `OmniAvatar/models/wan_video_dit.py:307-318` — audio_proj + audio_cond_projs setup
- `OmniAvatar/models/wan_video_dit.py:350-403` — forward pass: x+y concat, audio injection
- `OmniAvatar/models/wan_video_vae.py:548-573` — VAE encode (causal chunking)
- `OmniAvatar/models/wan_video_vae.py:614-621` — clear_cache
- `OmniAvatar/utils/io_utils.py:102-125` — smart_load_weights
- `OmniAvatar/scripts/train.py:365-371` — prepare_reference_input
- `OmniAvatar/scripts/train.py:549-555` — training forward (ref extraction)
- `StableAvatar/precompute_vae_latents.py:144-146` — masked video construction
- `StableAvatar/precompute_ref_latents.py:74-78` — reference segment selection
