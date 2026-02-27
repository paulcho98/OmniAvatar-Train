"""OmniAvatar V2V (Video-to-Video) Lip Sync Training Script.

Variant of train.py that regenerates only the mouth/lip region conditioned on:
- Spatial mask (LatentSync style) indicating mouth region to regenerate
- Masked source video (upper face preserved, mouth erased)
- Audio driving signal

Key difference from I2V train.py:
- in_dim = 49 (16 noise + 16 ref_repeated + 1 spatial_mask + 16 masked_video)
- Spatial mask replaces binary temporal mask
- Supports precomputed VAE latents and audio embeddings from StableAvatar pipeline

Usage:
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate_2gpus.yaml \\
        scripts/train_v2v.py \\
        --dit_paths "pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-0000{1..6}-of-00006.safetensors" \\
        --text_encoder_path pretrained_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth \\
        --vae_path pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \\
        --wav2vec_path pretrained_models/wav2vec2-base-960h \\
        --omniavatar_ckpt pretrained_models/OmniAvatar-14B/pytorch_model.pt \\
        --data_list_path /path/to/video_square_path.txt \\
        --latentsync_mask_path /path/to/mask.png \\
        --use_precomputed_vae --use_precomputed_audio \\
        --use_gradient_checkpointing --num_frames 81 --height 512 --width 512
"""

import os
import sys
import json
import math
import random
import argparse
import subprocess
import tempfile
import shutil
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import imageio
from tqdm import tqdm
from PIL import Image
from peft import LoraConfig, inject_adapter_in_model
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
import torchvision.transforms as transforms
from transformers import Wav2Vec2FeatureExtractor

# ---------------------------------------------------------------------------
# Step 1: Args singleton setup — MUST happen before any OmniAvatar imports.
# V2V uses in_dim=49: 16ch noise + 16ch ref + 1ch spatial_mask + 16ch masked_video
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import OmniAvatar.utils.args_config as args_module


def setup_omniavatar_args():
    """Configure the global args singleton for V2V (49-channel input)."""
    args_obj = argparse.Namespace(
        use_audio=True,
        sp_size=1,           # No sequence parallel — Accelerate handles DDP
        model_config={"in_dim": 49, "audio_hidden_size": 32},
        i2v=True,
        random_prefix_frames=True,
    )
    args_module.args = args_obj
    return args_obj


# Set up before any OmniAvatar model imports
setup_omniavatar_args()

# ---------------------------------------------------------------------------
# Step 2: OmniAvatar imports (safe now that args singleton is configured)
# ---------------------------------------------------------------------------
from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.models.wav2vec import Wav2VecModel
from OmniAvatar.wan_video import WanVideoPipeline
from OmniAvatar.utils.io_utils import load_state_dict


# ============================================================================
# Dataset
# ============================================================================

class OmniAvatarV2VDataset(torch.utils.data.Dataset):
    """Dataset for OmniAvatar V2V training.

    Reads video directories from a text file (one per line). Each directory contains:
    - sub_clip.mp4: Source video
    - audio.wav: Extracted 16kHz mono audio
    - prompt.txt: Text prompt
    - vae_latents.pt: {input_latents, masked_latents, reference_frame, metadata}
    - audio_emb_omniavatar.pt: {audio_emb: [total_frames, 10752], metadata}

    Supports precomputed VAE latents and audio embeddings for fast training.
    """

    def __init__(self, data_list_path, num_frames=81, height=512, width=512,
                 sample_rate=16000, fps=25, repeat=1,
                 latentsync_mask_path=None,
                 use_precomputed_vae=False, use_precomputed_audio=False,
                 use_precomputed_text_emb=False):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.sample_rate = sample_rate
        self.fps = fps
        self.repeat = repeat
        self.use_precomputed_vae = use_precomputed_vae
        self.use_precomputed_audio = use_precomputed_audio
        self.use_precomputed_text_emb = use_precomputed_text_emb

        # Read video directories
        with open(data_list_path) as f:
            self.video_dirs = [line.strip() for line in f if line.strip()]

        # Load LatentSync mask (1=keep, 0=mask in source; we store as-is)
        self.latentsync_mask = None
        if latentsync_mask_path and os.path.exists(latentsync_mask_path):
            mask_img = Image.open(latentsync_mask_path)
            mask_arr = np.array(mask_img)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]  # Take first channel
            # Normalize to [0, 1]: 255→1 (keep), 0→0 (mask)
            self.latentsync_mask = torch.from_numpy(mask_arr.astype(np.float32) / 255.0)

    def _getitem_precomputed(self, video_dir):
        """Load precomputed VAE latents and audio embeddings."""
        result = {}

        # VAE latents
        if self.use_precomputed_vae:
            vae_path = os.path.join(video_dir, "vae_latents.pt")
            vae_data = torch.load(vae_path, map_location="cpu")
            result["precomputed_input_latents"] = vae_data["input_latents"]    # [16, 21, 64, 64]
            result["precomputed_masked_latents"] = vae_data["masked_latents"]  # [16, 21, 64, 64]

        # Audio embeddings (OmniAvatar format)
        if self.use_precomputed_audio:
            audio_path = os.path.join(video_dir, "audio_emb_omniavatar.pt")
            audio_data = torch.load(audio_path, map_location="cpu")
            result["precomputed_audio_emb"] = audio_data["audio_emb"]  # [total_frames, 10752]

        # Precomputed text embeddings
        if self.use_precomputed_text_emb:
            text_emb_path = os.path.join(video_dir, "text_emb.pt")
            if os.path.exists(text_emb_path):
                result["precomputed_text_emb"] = torch.load(
                    text_emb_path, map_location="cpu", weights_only=False
                )
            else:
                print(f"[Dataset] WARNING: text_emb.pt missing in {video_dir}, falling back to live T5 encoding")

        # Prompt (still needed as fallback or for text dropout empty encoding init)
        prompt_path = os.path.join(video_dir, "prompt.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path) as f:
                result["prompt"] = f.read().strip()
        else:
            result["prompt"] = "a person is talking"

        # Raw audio for aux losses (loaded here by DataLoader workers, not in forward())
        audio_wav_path = os.path.join(video_dir, "audio.wav")
        if os.path.exists(audio_wav_path):
            result["audio_path"] = audio_wav_path
            audio_np, _ = librosa.load(audio_wav_path, sr=self.sample_rate)
            target_samples = int(self.num_frames / 25 * self.sample_rate)
            if len(audio_np) > target_samples:
                audio_np = audio_np[:target_samples]
            result["audio"] = audio_np

        result["video_dir"] = video_dir
        return result

    def _crop_and_resize(self, image, target_h, target_w):
        """Center crop and resize to target dimensions."""
        w, h = image.size
        scale = max(target_w / w, target_h / h)
        image = transforms.functional.resize(
            image, (round(h * scale), round(w * scale)),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        return transforms.functional.center_crop(image, (target_h, target_w))

    def _load_video(self, file_path):
        """Load video frames as PIL Images."""
        reader = imageio.get_reader(file_path)
        num_available = reader.count_frames()
        num_frames = min(self.num_frames, num_available)
        while num_frames > 1 and num_frames % 4 != 1:
            num_frames -= 1
        frames = []
        for i in range(num_frames):
            frame = Image.fromarray(reader.get_data(i))
            frame = self._crop_and_resize(frame, self.height, self.width)
            frames.append(frame)
        reader.close()
        return frames

    def _apply_spatial_mask(self, frames):
        """Apply LatentSync mask: frame 0 untouched, frames 1+ have mouth zeroed out."""
        if self.latentsync_mask is None:
            return frames

        masked_frames = [frames[0]]  # Frame 0: full reference, untouched
        mask = self.latentsync_mask  # [H_mask, W_mask], 1=keep, 0=mask

        for frame in frames[1:]:
            frame_np = np.array(frame).astype(np.float32)
            h, w = frame_np.shape[:2]
            # Resize mask to frame resolution if needed
            if mask.shape[0] != h or mask.shape[1] != w:
                mask_resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0), size=(h, w),
                    mode="bilinear", align_corners=False,
                )
                mask_np = (mask_resized.squeeze().numpy() > 0.5).astype(np.float32)
            else:
                mask_np = (mask.numpy() > 0.5).astype(np.float32)
            # Apply: keep upper face (1), zero out mouth (0)
            frame_np = frame_np * mask_np[:, :, None]
            masked_frames.append(Image.fromarray(frame_np.astype(np.uint8)))

        return masked_frames

    def _getitem_live(self, video_dir):
        """Load video and audio from files, apply mask live."""
        video_path = os.path.join(video_dir, "sub_clip.mp4")
        audio_path = os.path.join(video_dir, "audio.wav")

        frames = self._load_video(video_path)
        if len(frames) < 5:
            return None

        masked_frames = self._apply_spatial_mask(frames)

        # Load audio
        audio_np, _ = librosa.load(audio_path, sr=self.sample_rate)
        num_frames = len(frames)
        target_samples = int(num_frames / self.fps * self.sample_rate)
        if len(audio_np) > target_samples:
            audio_np = audio_np[:target_samples]

        # Prompt
        prompt_path = os.path.join(video_dir, "prompt.txt")
        prompt = "a person is talking"
        if os.path.exists(prompt_path):
            with open(prompt_path) as f:
                prompt = f.read().strip()

        return {
            "video": frames,
            "masked_video": masked_frames,
            "audio": audio_np,
            "prompt": prompt,
            "ref_image": frames[0],
            "video_dir": video_dir,
            "audio_path": audio_path,
        }

    def __getitem__(self, data_id):
        video_dir = self.video_dirs[data_id % len(self.video_dirs)]
        try:
            if self.use_precomputed_vae or self.use_precomputed_audio or self.use_precomputed_text_emb:
                # Use precomputed path — can mix precomputed VAE/audio/text
                result = self._getitem_precomputed(video_dir)
                # If VAE not precomputed, load video live for forward() to encode
                if not self.use_precomputed_vae:
                    video_path = os.path.join(video_dir, "sub_clip.mp4")
                    frames = self._load_video(video_path)
                    if len(frames) < 5:
                        return None
                    result["video"] = frames
                    result["masked_video"] = self._apply_spatial_mask(frames)
                    result["ref_image"] = frames[0]
                # If audio not precomputed, load raw audio for forward() to encode
                if not self.use_precomputed_audio:
                    audio_path = os.path.join(video_dir, "audio.wav")
                    audio_np, _ = librosa.load(audio_path, sr=self.sample_rate)
                    result["audio"] = audio_np
                return result
            else:
                return self._getitem_live(video_dir)
        except Exception as e:
            print(f"[Dataset] Error loading {video_dir}: {e}")
            return None

    def __len__(self):
        return len(self.video_dirs) * self.repeat


# ============================================================================
# Training Module
# ============================================================================

class OmniAvatarV2VTrainingModule(nn.Module):
    """Training module for OmniAvatar V2V lip sync.

    49-channel input: 16ch noise + 16ch ref_repeated + 1ch spatial_mask + 16ch masked_video.

    Frozen: T5 text encoder, VAE, Wav2Vec2
    Trainable (LoRA): DiT attention (q,k,v,o) and FFN (ffn.0, ffn.2)
    Trainable (full): AudioPack, audio_cond_projs, patch_embedding
    """

    def __init__(self, dit_paths, text_encoder_path, vae_path, wav2vec_path,
                 omniavatar_ckpt=None, lora_rank=128, lora_alpha=64,
                 lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                 use_gradient_checkpointing=True,
                 use_gradient_checkpointing_offload=False,
                 latentsync_mask=None,
                 num_training_frames=81,
                 dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.latentsync_mask = latentsync_mask  # [H_mask, W_mask], 1=keep, 0=mask
        self.num_training_frames = num_training_frames

        # --- 1. Load base models via ModelManager ---
        model_manager = ModelManager(device="cpu", infer=False)
        model_manager.load_models(
            [dit_paths.split(","), text_encoder_path, vae_path],
            torch_dtype=dtype, device="cpu",
        )

        # --- 2. Create pipeline ---
        self.pipe = WanVideoPipeline.from_model_manager(
            model_manager, torch_dtype=dtype, device="cpu",
        )

        # --- 3. Set scheduler to training mode ---
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # --- 4. Freeze everything ---
        self.pipe.requires_grad_(False)

        # --- 5. Add LoRA to DiT ---
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha,
            init_lora_weights=True,
            target_modules=lora_target_modules.split(","),
        )
        self.pipe.dit = inject_adapter_in_model(lora_config, self.pipe.dit)
        for param in self.pipe.dit.parameters():
            if param.requires_grad:
                param.data = param.to(dtype)

        # --- 6. Load OmniAvatar checkpoint (33ch → 49ch expansion) ---
        if omniavatar_ckpt:
            print(f"[Train] Loading OmniAvatar checkpoint: {omniavatar_ckpt}")
            ckpt_sd = load_state_dict(omniavatar_ckpt)
            # Map LoRA keys: lora_A.weight → lora_A.default.weight
            mapped_sd = {}
            for k, v in ckpt_sd.items():
                new_k = k
                if "lora_A.weight" in k:
                    new_k = k.replace("lora_A.weight", "lora_A.default.weight")
                if "lora_B.weight" in k:
                    new_k = k.replace("lora_B.weight", "lora_B.default.weight")
                mapped_sd[new_k] = v

            # Handle patch_embedding expansion: 33ch → 49ch
            pe_key = "patch_embedding.weight"
            if pe_key in mapped_sd:
                model_pe = self.pipe.dit.patch_embedding.weight
                if mapped_sd[pe_key].shape != model_pe.shape:
                    print(f"[Train] Expanding patch_embedding: {mapped_sd[pe_key].shape} → {model_pe.shape}")
                    new_pe = model_pe.data.clone()
                    slices = tuple(slice(0, s) for s in mapped_sd[pe_key].shape)
                    new_pe[slices] = mapped_sd[pe_key]
                    mapped_sd[pe_key] = new_pe

            pe_bias_key = "patch_embedding.bias"
            if pe_bias_key in mapped_sd and self.pipe.dit.patch_embedding.bias is not None:
                model_pe_bias = self.pipe.dit.patch_embedding.bias
                if mapped_sd[pe_bias_key].shape != model_pe_bias.shape:
                    new_bias = model_pe_bias.data.clone()
                    new_bias[:mapped_sd[pe_bias_key].shape[0]] = mapped_sd[pe_bias_key]
                    mapped_sd[pe_bias_key] = new_bias

            missing, unexpected = self.pipe.dit.load_state_dict(mapped_sd, strict=False)
            loaded = len(ckpt_sd) - len(unexpected)
            print(f"[Train] Loaded {loaded} params, {len(missing)} missing, {len(unexpected)} unexpected")

        # --- 7. Unfreeze audio modules ---
        if hasattr(self.pipe.dit, "audio_proj"):
            for p in self.pipe.dit.audio_proj.parameters():
                p.requires_grad = True
        if hasattr(self.pipe.dit, "audio_cond_projs"):
            for p in self.pipe.dit.audio_cond_projs.parameters():
                p.requires_grad = True

        # --- 8. Unfreeze patch_embedding ---
        self.pipe.dit.patch_embedding.weight.requires_grad = True
        if self.pipe.dit.patch_embedding.bias is not None:
            self.pipe.dit.patch_embedding.bias.requires_grad = True

        # --- 9. Load Wav2Vec2 (frozen, float32 matching inference) ---
        self.wav2vec = Wav2VecModel.from_pretrained(
            wav2vec_path, local_files_only=True,
        )
        self.wav2vec.eval()
        self.wav2vec.requires_grad_(False)
        self.wav2vec.feature_extractor._freeze_parameters()
        self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path)

        # Convenience references
        self.dit = self.pipe.dit
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.scheduler = self.pipe.scheduler
        self.prompter = self.pipe.prompter

        # Condition dropout
        self.text_drop_prob = 0.0
        self.audio_drop_prob = 0.0
        self._empty_text_cache = None

        # Auxiliary losses
        self.syncnet = None
        self.lpips_func = None
        self.trepa_func = None
        self._last_aux_losses = None

        # VRAM management
        self.offload_frozen = False
        self.verbose_vram = False

        # Explicit train/eval modes (DiffSynth freeze_except pattern)
        self._set_training_modes()

    def _set_training_modes(self):
        """Set train/eval modes explicitly following DiffSynth's freeze_except pattern."""
        self.pipe.dit.train()
        self.pipe.text_encoder.eval()
        self.pipe.vae.eval()
        self.wav2vec.eval()
        if self.syncnet is not None:
            self.syncnet.eval()
        if self.lpips_func is not None:
            self.lpips_func.eval()
        if self.trepa_func is not None:
            # TREPALoss is not an nn.Module, but its internal model is
            if hasattr(self.trepa_func, 'model'):
                self.trepa_func.model.eval()
            elif hasattr(self.trepa_func, 'eval'):
                self.trepa_func.eval()

    def _log_vram(self, label):
        """Log VRAM usage if verbose_vram is enabled."""
        if not self.verbose_vram:
            return
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [VRAM {label}] alloc={alloc:.2f}GB peak={peak:.2f}GB")

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

    # ------------------------------------------------------------------
    # Encoding helpers (all under no_grad — frozen encoders)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_audio(self, audio_np, num_video_frames, device):
        """Encode raw waveform → audio embeddings. Matches inference.py:238-256."""
        input_values = np.squeeze(
            self.wav_feature_extractor(audio_np, sampling_rate=16000).input_values
        )
        input_values = torch.from_numpy(input_values).float().to(device=device)
        input_values = input_values.unsqueeze(0)

        samples_per_frame = self.wav_feature_extractor.sampling_rate // 25
        target_samples = num_video_frames * samples_per_frame
        if input_values.shape[1] < target_samples:
            input_values = F.pad(input_values, (0, target_samples - input_values.shape[1]))

        hidden_states = self.wav2vec(
            input_values, seq_len=num_video_frames, output_hidden_states=True,
        )
        audio_emb = hidden_states.last_hidden_state
        for hs in hidden_states.hidden_states:
            audio_emb = torch.cat((audio_emb, hs), -1)
        return audio_emb  # (1, num_video_frames, 10752)

    @torch.no_grad()
    def encode_text(self, prompt, device):
        self.text_encoder.to(device)
        prompt_emb = self.prompter.encode_prompt(prompt, positive=True, device=device)
        return prompt_emb

    @torch.no_grad()
    def encode_video(self, video_frames, device):
        self.vae.to(device)
        frames_np = [np.array(f, dtype=np.float32) * (2.0 / 255.0) - 1.0 for f in video_frames]
        frames_t = [torch.from_numpy(f).permute(2, 0, 1) for f in frames_np]
        video_tensor = torch.stack(frames_t, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        latents = self.vae.encode(
            video_tensor.to(dtype=self.dtype, device=device),
            device=device, tiled=True,
            tile_size=(34, 34), tile_stride=(18, 16),
        )
        return latents.to(dtype=self.dtype, device=device)

    # ------------------------------------------------------------------
    # V2V-specific input preparation
    # ------------------------------------------------------------------

    def _get_latent_resolution_mask(self, latents):
        """Resize LatentSync mask to latent resolution.

        Args:
            latents: [B, C, T, H_lat, W_lat]
        Returns:
            mask: [H_lat, W_lat] float tensor (1=keep, 0=mask)
        """
        if self.latentsync_mask is None:
            return None
        mask = self.latentsync_mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
        H_lat, W_lat = latents.shape[3], latents.shape[4]
        mask_resized = F.interpolate(mask, size=(H_lat, W_lat), mode="bilinear", align_corners=False)
        return (mask_resized > 0.5).float().squeeze(0).squeeze(0)  # [H_lat, W_lat]

    def prepare_v2v_input(self, ref_latent, masked_video_latents, latent_mask):
        """Prepare 33-channel y tensor for V2V: ref_repeated + spatial_mask + masked_video.

        Note: The full 49ch input to the DiT is x (16ch noise) cat y (33ch), done in the DiT's
        forward pass at wan_video_dit.py:357.

        Args:
            ref_latent: [1, 16, 1, H_lat, W_lat]
            masked_video_latents: [1, 16, T_lat, H_lat, W_lat]
            latent_mask: [H_lat, W_lat] (LatentSync: 1=keep, 0=mask)
        Returns:
            y: [1, 33, T_lat, H_lat, W_lat]
        """
        T_lat = masked_video_latents.shape[2]
        H_lat, W_lat = masked_video_latents.shape[3], masked_video_latents.shape[4]
        device = masked_video_latents.device
        dtype = masked_video_latents.dtype

        ref_repeated = ref_latent.repeat(1, 1, T_lat, 1, 1)  # [1, 16, T, H, W]

        # OmniAvatar mask convention: 0=keep, 1=generate
        # Invert LatentSync (1=keep→0, 0=mask→1): mouth becomes 1 (generate)
        inverted = 1.0 - latent_mask  # [H_lat, W_lat]
        mask_ch = torch.zeros(1, 1, T_lat, H_lat, W_lat, device=device, dtype=dtype)
        mask_ch[:, :, 0] = 0  # Frame 0: keep all (reference frame)
        mask_ch[:, :, 1:] = inverted[None, None, None]  # Frames 1+: spatial mask

        return torch.cat([ref_repeated, mask_ch, masked_video_latents], dim=1)  # [1, 33, T, H, W]

    # ------------------------------------------------------------------
    # Auxiliary losses (identical to train.py)
    # ------------------------------------------------------------------

    def setup_aux_losses(self, args):
        self.aux_recon_weight = args.aux_recon_weight
        self.aux_sync_weight = args.aux_sync_weight
        self.aux_lpips_weight = args.aux_lpips_weight
        self.aux_trepa_weight = args.aux_trepa_weight
        self.aux_num_frames = args.aux_num_frames
        self.sync_chunk_size = args.sync_chunk_size
        self.sync_chunk_stride = args.sync_chunk_stride
        self.sync_num_supervised_frames = args.sync_num_supervised_frames

        if args.use_vae_gradient_checkpointing:
            self.pipe.vae.model.decoder.gradient_checkpointing = True
            print("[AuxLoss] Enabled VAE decoder gradient checkpointing")

        latentsync_path = "/home/work/.local/Self-Forcing_LipSync_StableAvatar"

        if args.use_sync_loss:
            sys.path.insert(0, latentsync_path)
            from latentsync_models.stable_syncnet import StableSyncNet
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(args.syncnet_config_path)
            self.syncnet = StableSyncNet(
                OmegaConf.to_container(cfg.model), gradient_checkpointing=True
            )
            ckpt = torch.load(args.syncnet_checkpoint_path, map_location="cpu")
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            self.syncnet.load_state_dict(ckpt, strict=True)
            self.syncnet.requires_grad_(False)
            self.syncnet.eval()
            self.syncnet_config = cfg
            from latentsync_models.audio import melspectrogram as _mel_fn
            self._melspectrogram = _mel_fn
            print(f"[AuxLoss] Loaded SyncNet from {args.syncnet_checkpoint_path}")

        if args.use_lpips_loss:
            import lpips as lpips_pkg
            self.lpips_func = lpips_pkg.LPIPS(net="vgg")
            self.lpips_func.requires_grad_(False)
            self.lpips_func.eval()
            print("[AuxLoss] Loaded LPIPS (VGG)")

        if args.use_trepa_loss:
            sys.path.insert(0, latentsync_path)
            from latentsync_models.trepa.loss import TREPALoss
            self.trepa_func = TREPALoss(
                device="cpu", ckpt_path=args.trepa_checkpoint_path, with_cp=True
            )
            print(f"[AuxLoss] Loaded TREPA from {args.trepa_checkpoint_path}")

    def _decode_for_aux_loss(self, latents, device, with_grad=True):
        self.pipe.vae.model.to(device)
        latents = latents.to(device=device, dtype=self.pipe.vae.model.conv2.weight.dtype)
        if with_grad:
            rgb = self.pipe.vae.model.decode(latents, self.pipe.vae.scale)
            return rgb.clamp(-1, 1)
        else:
            with torch.no_grad():
                rgb = self.pipe.vae.model.decode(latents, self.pipe.vae.scale)
                return rgb.clamp_(-1, 1)

    def _compute_sync_loss(self, pred_rgb, audio_np, device):
        from einops import rearrange

        B, C, T, H, W = pred_rgb.shape
        chunk_size = self.sync_chunk_size
        stride = self.sync_chunk_stride
        num_supervised = self.sync_num_supervised_frames
        fps = 25
        sr = 16000

        pred_flat = rearrange(pred_rgb, "b c t h w -> (b t) c h w")
        pred_resized = F.interpolate(
            pred_flat, size=(256, 256), mode="bicubic", align_corners=False
        )
        pred_lower = pred_resized[:, :, 128:, :]

        chunk_losses = []
        num_chunks = max(1, math.ceil((min(T, num_supervised) - chunk_size) / stride) + 1)

        for i in range(num_chunks):
            start = i * stride
            end = min(start + chunk_size, T)
            if end - start < chunk_size:
                continue

            try:
                chunk_frames = pred_lower[start:end]
                vis_input = rearrange(chunk_frames, "(b t) c h w -> b (t c) h w", b=B)

                start_sample = int(start / fps * sr)
                end_sample = int(end / fps * sr)
                audio_seg = audio_np[start_sample:end_sample]
                if len(audio_seg) < 100:
                    audio_seg = np.zeros(int((end - start) / fps * sr), dtype=np.float32)
                mel = self._melspectrogram(audio_seg)
                mel_window = math.ceil((end - start) / 5.0 * 16)
                mel_t = torch.from_numpy(mel).float()
                mel_t = F.pad(mel_t, (0, max(0, mel_window - mel_t.shape[1])))[:, :mel_window]
                mel_t = mel_t.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float16)

                v_emb, a_emb = self.syncnet(vis_input.half(), mel_t)
                sims = F.cosine_similarity(v_emb, a_emb, dim=1).unsqueeze(1)
                loss = F.binary_cross_entropy_with_logits(
                    sims, torch.ones_like(sims)
                ).mean()
                chunk_losses.append(loss)
            except Exception as e:
                print(f"[SyncLoss] Warning: chunk {i} (frames {start}-{end}) failed: {e}")
                continue

        if not chunk_losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(chunk_losses).mean()

    def _compute_lpips_loss(self, pred_rgb, gt_rgb):
        from einops import rearrange

        pred_flat = rearrange(pred_rgb, "b c t h w -> (b t) c h w")
        gt_flat = rearrange(gt_rgb, "b c t h w -> (b t) c h w")
        H = pred_flat.shape[2]
        return self.lpips_func(
            pred_flat[:, :, H // 2:, :].float(),
            gt_flat[:, :, H // 2:, :].float(),
        ).mean()

    def _compute_trepa_loss(self, pred_rgb, gt_rgb):
        return self.trepa_func(pred_rgb, gt_rgb)

    # ------------------------------------------------------------------
    # Forward pass — flow matching training loss
    # ------------------------------------------------------------------

    def _load_raw_audio_for_aux(self, data, device):
        """Load raw audio numpy array for auxiliary losses (SyncNet)."""
        if "audio" in data:
            return data["audio"]
        if "audio_path" in data:
            audio_np, _ = librosa.load(data["audio_path"], sr=16000)
            # Trim to training frame duration
            target_samples = int(self.num_training_frames / 25 * 16000)
            if len(audio_np) > target_samples:
                audio_np = audio_np[:target_samples]
            return audio_np
        return None

    def forward(self, data):
        """Training forward pass with V2V input construction."""
        device = next(self.dit.parameters()).device

        # Defensive: ensure DiT is in training mode for gradient checkpointing
        if not self.dit.training:
            self.dit.train()
        self._log_vram("forward-start")

        # 1. Encode text (precomputed or live)
        if "precomputed_text_emb" in data:
            context = data["precomputed_text_emb"].to(device, dtype=self.dtype)
            # Ensure shape is [1, 512, 4096] — keep full padded length to match live path
            if context.dim() == 2:
                context = context.unsqueeze(0)
        else:
            if self.offload_frozen:
                self.text_encoder.to(device)
            context = self.encode_text(data["prompt"], device)
            if self.offload_frozen:
                self.text_encoder.to("cpu")
                torch.cuda.empty_cache()
        self._log_vram("after-text-encode")

        # 2. Encode audio (precomputed or live)
        if "precomputed_audio_emb" in data:
            full_emb = data["precomputed_audio_emb"]  # [total_frames, 10752]
            audio_emb = full_emb[:self.num_training_frames]  # Slice to training frames
            audio_emb = audio_emb.unsqueeze(0).to(dtype=self.dtype, device=device)
        else:
            if self.offload_frozen:
                self.wav2vec.to(device)
            num_video_frames = len(data["video"])
            audio_emb = self.encode_audio(data["audio"], num_video_frames, device)
            audio_emb = audio_emb.to(dtype=self.dtype, device=device)
            if self.offload_frozen:
                self.wav2vec.to("cpu")
                torch.cuda.empty_cache()
        self._log_vram("after-audio-encode")

        # 3. Condition dropout
        if self.training:
            if self.text_drop_prob > 0 and random.random() < self.text_drop_prob:
                if self._empty_text_cache is None:
                    if self.offload_frozen:
                        self.text_encoder.to(device)
                    self._empty_text_cache = self.encode_text("", device)
                    if self.offload_frozen:
                        self.text_encoder.to("cpu")
                        torch.cuda.empty_cache()
                context = self._empty_text_cache.clone()
            if self.audio_drop_prob > 0 and random.random() < self.audio_drop_prob:
                audio_emb = torch.zeros_like(audio_emb)

        # 4. Encode video (precomputed or live)
        if "precomputed_input_latents" in data:
            input_latents = data["precomputed_input_latents"].unsqueeze(0).to(
                dtype=self.dtype, device=device
            )
            masked_video_latents = data["precomputed_masked_latents"].unsqueeze(0).to(
                dtype=self.dtype, device=device
            )
        else:
            # VAE always on GPU (only 0.26 GB, needed for aux loss backward)
            input_latents = self.encode_video(data["video"], device)
            masked_video_latents = self.encode_video(data["masked_video"], device)
        self._log_vram("after-vae-encode")

        # 5. Prepare V2V input (y = ref_repeated + spatial_mask + masked_video)
        ref_latent = input_latents[:, :, :1]
        latent_mask = self._get_latent_resolution_mask(input_latents)
        if latent_mask is not None:
            latent_mask = latent_mask.to(device=device)
        else:
            H_lat, W_lat = input_latents.shape[3], input_latents.shape[4]
            latent_mask = torch.ones(H_lat, W_lat, device=device)
        y = self.prepare_v2v_input(ref_latent, masked_video_latents, latent_mask)

        # 6. Sample noise + timestep
        noise = torch.randn_like(input_latents)
        num_timesteps = len(self.scheduler.timesteps)
        timestep_id = torch.randint(0, num_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=device)

        noisy_latents = self.scheduler.add_noise(input_latents, noise, timestep)
        training_target = self.scheduler.training_target(input_latents, noise, timestep)
        self._log_vram("before-dit-forward")

        # 7. DiT forward
        noise_pred = self.dit(
            x=noisy_latents, timestep=timestep, context=context, y=y,
            audio_emb=audio_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )
        self._log_vram("after-dit-forward")

        # 8. Flow matching MSE loss with timestep weighting
        # Apply timestep weight to MSE ONLY (not aux losses) — matching StableAvatar
        loss = F.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)

        # 9. Auxiliary losses on decoded x_0
        T_lat = input_latents.shape[2]
        use_any_aux = (self.syncnet is not None or self.lpips_func is not None
                       or self.trepa_func is not None)
        if use_any_aux:
            self._log_vram("before-aux-losses")
            mse_loss_val = loss.detach()  # Save before combining

            sigma = self.scheduler.sigmas[timestep_id].to(
                device=device, dtype=noise_pred.dtype
            )
            sync_len = self.aux_num_frames if self.aux_num_frames > 0 else T_lat

            x_0_pred = noisy_latents[:, :, :sync_len] - sigma * noise_pred[:, :, :sync_len]
            x_0_pred = torch.clamp(x_0_pred, -10, 10)

            # VAE always on GPU (0.26 GB) — backward needs it for gradient flow
            pred_rgb = self._decode_for_aux_loss(x_0_pred, device, with_grad=True)
            self._log_vram("after-vae-decode-pred")
            gt_rgb = self._decode_for_aux_loss(
                input_latents[:, :, :sync_len].detach(), device, with_grad=False
            )
            self._log_vram("after-vae-decode-gt")

            sync_loss = torch.tensor(0.0, device=device)
            lpips_loss = torch.tensor(0.0, device=device)
            trepa_loss = torch.tensor(0.0, device=device)

            if self.syncnet is not None:
                raw_audio = self._load_raw_audio_for_aux(data, device)
                if raw_audio is not None:
                    sync_loss = self._compute_sync_loss(pred_rgb, raw_audio, device)
            if self.lpips_func is not None:
                lpips_loss = self._compute_lpips_loss(pred_rgb, gt_rgb)
            if self.trepa_func is not None and pred_rgb.shape[2] >= 16:
                trepa_loss = self._compute_trepa_loss(pred_rgb, gt_rgb)

            # Combine: timestep-weighted MSE + unweighted aux losses
            loss = (
                loss * self.aux_recon_weight
                + sync_loss * self.aux_sync_weight
                + lpips_loss * self.aux_lpips_weight
                + trepa_loss * self.aux_trepa_weight
            )

            # Store as detached tensors — .item() deferred to logging time
            self._last_aux_losses = {
                "mse": mse_loss_val,
                "sync": sync_loss.detach(),
                "lpips": lpips_loss.detach(),
                "trepa": trepa_loss.detach(),
                "total": loss.detach(),
            }

        return loss


# ============================================================================
# V2V Validation
# ============================================================================

def load_v2v_validation_data(val_dir, max_samples=4):
    """Load precomputed validation samples from a directory of video dirs.

    Each video dir should have:
    - vae_latents.pt: {input_latents, masked_latents, ...}
    - audio_emb_omniavatar.pt: {audio_emb: [total_frames, 10752]}
    - prompt.txt
    - audio.wav (for muxing output video)
    """
    path_file = os.path.join(val_dir, "video_square_path.txt")
    if not os.path.exists(path_file):
        print(f"[Validation] No video_square_path.txt found in {val_dir}")
        return []

    with open(path_file) as f:
        video_dirs = [line.strip() for line in f if line.strip()]

    samples = []
    for vdir in video_dirs[:max_samples]:
        try:
            sample = {}

            # VAE latents
            vae_path = os.path.join(vdir, "vae_latents.pt")
            if not os.path.exists(vae_path):
                print(f"[Validation] Missing vae_latents.pt in {vdir}")
                continue
            vae_data = torch.load(vae_path, map_location="cpu")
            sample["input_latents"] = vae_data["input_latents"]      # [16, 21, 64, 64]
            sample["masked_latents"] = vae_data["masked_latents"]    # [16, 21, 64, 64]

            # Audio embeddings
            audio_path = os.path.join(vdir, "audio_emb_omniavatar.pt")
            if not os.path.exists(audio_path):
                print(f"[Validation] Missing audio_emb_omniavatar.pt in {vdir}")
                continue
            audio_data = torch.load(audio_path, map_location="cpu")
            sample["audio_emb"] = audio_data["audio_emb"]  # [total_frames, 10752]

            # Prompt
            prompt_path = os.path.join(vdir, "prompt.txt")
            if os.path.exists(prompt_path):
                with open(prompt_path) as f:
                    sample["prompt"] = f.read().strip()
            else:
                sample["prompt"] = "a person is talking"

            # Audio wav path (for muxing)
            wav_path = os.path.join(vdir, "audio.wav")
            if os.path.exists(wav_path):
                sample["audio_wav_path"] = wav_path

            sample["video_dir"] = vdir
            sample["name"] = os.path.basename(vdir)
            samples.append(sample)

        except Exception as e:
            print(f"[Validation] Error loading {vdir}: {e}")

    print(f"[Validation] Loaded {len(samples)} samples from {val_dir}")
    return samples


@torch.no_grad()
def run_v2v_validation(model, sample, device, num_training_frames=81,
                       num_inference_steps=25, guidance_scale=4.5, negative_prompt=""):
    """Generate V2V video from precomputed validation data.

    Uses masked video + ref frame + spatial mask as conditioning, denoises from noise.
    """
    model.eval()
    pipe = model.pipe

    # 1. Encode text (transient GPU load if offloading)
    if model.offload_frozen:
        model.text_encoder.to(device)
    context = model.encode_text(sample["prompt"], device)
    neg_context = model.encode_text(negative_prompt, device) if guidance_scale != 1.0 else None
    if model.offload_frozen:
        model.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    # 2. Audio embeddings (precomputed, slice to training frames)
    full_emb = sample["audio_emb"]
    audio_emb = full_emb[:num_training_frames].unsqueeze(0).to(dtype=model.dtype, device=device)

    # 3. Load precomputed latents
    input_latents = sample["input_latents"].unsqueeze(0).to(dtype=model.dtype, device=device)
    masked_video_latents = sample["masked_latents"].unsqueeze(0).to(dtype=model.dtype, device=device)

    # 4. Prepare V2V input
    ref_latent = input_latents[:, :, :1]
    T_lat = input_latents.shape[2]
    latent_mask = model._get_latent_resolution_mask(input_latents)
    if latent_mask is not None:
        latent_mask = latent_mask.to(device=device)
    else:
        H_lat, W_lat = input_latents.shape[3], input_latents.shape[4]
        latent_mask = torch.ones(H_lat, W_lat, device=device)
    y = model.prepare_v2v_input(ref_latent, masked_video_latents, latent_mask)

    # 5. Build "clean" latent for fixed-frame overwriting
    lat = torch.cat([
        ref_latent,
        torch.zeros(1, 16, T_lat - 1, ref_latent.shape[3], ref_latent.shape[4],
                     dtype=model.dtype, device=device),
    ], dim=2)

    # 6. Start from pure noise
    latents = torch.randn_like(lat)
    fixed_frame = 1

    pipe.scheduler.set_timesteps(num_inference_steps, shift=5.0)

    for ts in pipe.scheduler.timesteps:
        if fixed_frame > 0:
            latents[:, :, :fixed_frame] = lat[:, :, :fixed_frame]

        t = ts.unsqueeze(0).to(dtype=model.dtype, device=device)

        noise_pred_posi = model.dit(
            x=latents, timestep=t, context=context, y=y, audio_emb=audio_emb,
        )

        if guidance_scale != 1.0:
            audio_emb_uc = torch.zeros_like(audio_emb)
            noise_pred_nega = model.dit(
                x=latents, timestep=t, context=neg_context, y=y, audio_emb=audio_emb_uc,
            )
            noise_pred = noise_pred_nega + guidance_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi

        latents = pipe.scheduler.step(noise_pred, ts, latents)

    if fixed_frame > 0:
        latents[:, :, :fixed_frame] = lat[:, :, :fixed_frame]

    # Decode (VAE stays on GPU)
    old_device = pipe.device
    pipe.device = device
    frames = pipe.decode_video(latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
    pipe.device = old_device
    if frames.dim() == 5:
        frames = frames[0]
    video_frames = pipe.tensor2video(frames)

    # Restore scheduler and training modes
    pipe.scheduler.set_timesteps(1000, training=True)
    model._set_training_modes()
    return video_frames


def save_video_frames(frames, output_path, fps=25):
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()


def mux_video_with_audio(frames, audio_path, output_path, fps=25):
    temp_video = output_path.replace(".mp4", "_temp_nosound.mp4")
    save_video_frames(frames, temp_video, fps=fps)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_video, "-i", audio_path,
             "-c:v", "copy", "-c:a", "aac", "-shortest", output_path],
            capture_output=True, check=True,
        )
    except Exception:
        shutil.copy2(temp_video, output_path)
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)


def load_syncnet(args):
    if not getattr(args, "compute_sync_metrics", False):
        return None, None
    try:
        # SyncNet is in Self-Forcing_LipSync_StableAvatar, not DiffSynth-Studio
        diffsynth_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "Self-Forcing_LipSync_StableAvatar",
        )
        if diffsynth_path not in sys.path:
            sys.path.insert(0, diffsynth_path)
        from diffsynth.models.syncnet import SyncNetEval, SyncNetDetector
        syncnet = SyncNetEval(device="cuda" if torch.cuda.is_available() else "cpu")
        syncnet.loadParameters(args.syncnet_model_path)
        detector = SyncNetDetector(
            device="cuda" if torch.cuda.is_available() else "cpu",
            detect_results_dir=os.path.join(args.output_path, "syncnet_detect"),
            s3fd_model_path=args.s3fd_model_path,
        )
        print("[SyncNet] Loaded successfully")
        return syncnet, detector
    except Exception as e:
        print(f"[SyncNet] Failed to load: {e}")
        return None, None


def run_syncnet_eval(syncnet, detector, video_path, temp_dir):
    try:
        diffsynth_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "DiffSynth-Studio",
        )
        if diffsynth_path not in sys.path:
            sys.path.insert(0, diffsynth_path)
        from diffsynth.models.syncnet import syncnet_eval
        syncnet_temp = os.path.join(temp_dir, "syncnet_eval")
        os.makedirs(syncnet_temp, exist_ok=True)
        av_offset, sync_d, sync_c = syncnet_eval(
            syncnet, detector, video_path, syncnet_temp,
            detect_results_dir=detector.detect_results_dir,
        )
        shutil.rmtree(syncnet_temp, ignore_errors=True)
        return sync_d, sync_c
    except Exception as e:
        print(f"[SyncNet] Eval failed: {e}")
        return None, None


def run_v2v_validation_loop(model, val_samples, prefix, args, global_step, device,
                            syncnet=None, syncnet_detector=None, use_wandb=False):
    """Run V2V validation on preloaded samples, log to wandb with prefix."""
    unwrapped = model
    # Save validation videos persistently under output_path/val_videos/
    val_save_dir = os.path.join(args.output_path, "val_videos", f"step_{global_step}", prefix)
    os.makedirs(val_save_dir, exist_ok=True)
    val_temp_dir = tempfile.mkdtemp(prefix=f"val_v2v_{prefix}_")
    val_videos = []
    sync_d_list, sync_c_list = [], []

    for idx, sample in enumerate(val_samples):
        try:
            print(f"[Validation/{prefix}] Generating sample {idx + 1}/{len(val_samples)}: {sample['name']}...")
            generated_frames = run_v2v_validation(
                unwrapped, sample, device,
                num_training_frames=args.num_frames,
                num_inference_steps=args.val_num_inference_steps,
                guidance_scale=args.val_cfg_scale,
                negative_prompt=getattr(args, "negative_prompt", ""),
            )

            video_path = os.path.join(val_temp_dir, f"val_{prefix}_{global_step}_{idx}.mp4")
            audio_path = sample.get("audio_wav_path", "")

            if audio_path and os.path.exists(audio_path):
                mux_video_with_audio(generated_frames, audio_path, video_path, fps=25)
            else:
                save_video_frames(generated_frames, video_path, fps=25)

            # Save a persistent copy
            persistent_path = os.path.join(val_save_dir, f"{sample['name']}.mp4")
            shutil.copy2(video_path, persistent_path)

            # SyncNet metrics
            sync_d, sync_c = None, None
            if syncnet is not None and syncnet_detector is not None:
                sync_d, sync_c = run_syncnet_eval(syncnet, syncnet_detector, video_path, val_temp_dir)
                if sync_d is not None:
                    sync_d_list.append(sync_d)
                    sync_c_list.append(sync_c)
                    print(f"[SyncNet/{prefix}] Sample {idx}: Sync-D={sync_d:.2f}, Sync-C={sync_c:.2f}")

            if use_wandb:
                import wandb
                caption = f"{prefix}_{global_step}_{sample['name']}"
                if sync_d is not None:
                    caption += f"_D{sync_d:.1f}_C{sync_c:.1f}"
                val_videos.append(wandb.Video(video_path, format="mp4", caption=caption))

        except Exception as e:
            print(f"[Validation/{prefix}] Error on sample {idx}: {e}")
            import traceback
            traceback.print_exc()

    # Log to wandb
    if use_wandb and val_videos:
        try:
            import wandb
            wandb.log({f"val_{prefix}/videos": val_videos}, step=global_step)
            if sync_d_list:
                avg_d = sum(sync_d_list) / len(sync_d_list)
                avg_c = sum(sync_c_list) / len(sync_c_list)
                wandb.log({
                    f"val_{prefix}/sync_d": avg_d,
                    f"val_{prefix}/sync_c": avg_c,
                }, step=global_step)
                print(f"[Validation/{prefix}] Avg Sync-D: {avg_d:.2f}, Avg Sync-C: {avg_c:.2f}")
        except Exception as e:
            print(f"[Validation/{prefix}] wandb log error: {e}")

    shutil.rmtree(val_temp_dir, ignore_errors=True)
    print(f"[Validation/{prefix}] Videos saved to {val_save_dir}")


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(accelerator, model, output_path, step_or_name, train_args):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state_dict = accelerator.get_state_dict(model)
        unwrapped = accelerator.unwrap_model(model)
        state_dict = unwrapped.export_trainable_state_dict(
            state_dict, remove_prefix="pipe.dit.",
        )
        os.makedirs(output_path, exist_ok=True)
        path = os.path.join(output_path, f"step-{step_or_name}.pt")
        torch.save(state_dict, path)
        print(f"[Checkpoint] Saved {len(state_dict)} params to {path}")

        config = {
            "model_config": {"in_dim": 49, "audio_hidden_size": 32},
            "train_architecture": "lora",
            "lora_rank": train_args.lora_rank,
            "lora_alpha": train_args.lora_alpha,
            "lora_target_modules": train_args.lora_target_modules,
            "init_lora_weights": "kaiming",
            "use_audio": True,
            "i2v": True,
            "v2v": True,
            "random_prefix_frames": True,
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)


def save_training_state(accelerator, model, optimizer, lr_scheduler, ckpt_dir, train_args):
    """Save lean training state: trainable weights + optimizer + scheduler + RNG.

    Collective op — all ranks must call (barrier + per-rank RNG save).
    Replaces accelerator.save_state() to avoid saving the full 14B model (~45 GB → ~3.6 GB).
    """
    import pickle
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. Trainable model weights (main process only)
    if accelerator.is_main_process:
        state_dict = accelerator.get_state_dict(model)
        unwrapped = accelerator.unwrap_model(model)
        trainable_sd = unwrapped.export_trainable_state_dict(
            state_dict, remove_prefix="pipe.dit.",
        )
        torch.save(trainable_sd, os.path.join(ckpt_dir, "trainable_weights.pt"))
        print(f"[Checkpoint] Saved {len(trainable_sd)} trainable params "
              f"({sum(v.numel() for v in trainable_sd.values()) / 1e6:.1f}M parameters)")

    # 2. Optimizer state (main process only) — already only has trainable param states
    if accelerator.is_main_process:
        raw_opt = optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer
        torch.save(raw_opt.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

    # 3. LR scheduler state (main process only)
    if accelerator.is_main_process:
        raw_sched = lr_scheduler.scheduler if hasattr(lr_scheduler, 'scheduler') else lr_scheduler
        torch.save(raw_sched.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))

    # 4. RNG states (each rank saves its own)
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state(accelerator.device)
    with open(os.path.join(ckpt_dir, f"random_states_{accelerator.process_index}.pkl"), "wb") as f:
        pickle.dump(rng_state, f)

    # 5. Metadata (main process only)
    if accelerator.is_main_process:
        step_str = os.path.basename(ckpt_dir).split("-")[-1]
        metadata = {"format": "lean_v1", "global_step": int(step_str)}
        with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    # Barrier: all ranks finish writing before anyone proceeds
    accelerator.wait_for_everyone()


def load_training_state(accelerator, model, optimizer, lr_scheduler, ckpt_dir):
    """Load lean training state: trainable weights + optimizer + scheduler + RNG.

    Collective op — all ranks must call (per-rank RNG restore + barrier).
    Replaces accelerator.load_state().
    """
    import pickle

    # 1. Trainable model weights
    weights_path = os.path.join(ckpt_dir, "trainable_weights.pt")
    trainable_sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Re-add "pipe.dit." prefix stripped during save
    prefixed_sd = {f"pipe.dit.{k}": v for k, v in trainable_sd.items()}
    unwrapped = accelerator.unwrap_model(model)
    missing, unexpected = unwrapped.load_state_dict(prefixed_sd, strict=False)
    if accelerator.is_main_process:
        print(f"[Resume] Loaded {len(prefixed_sd)} trainable params from checkpoint")
        if unexpected:
            print(f"[Resume] WARNING: {len(unexpected)} unexpected keys")

    # 2. Optimizer state
    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    opt_sd = torch.load(opt_path, map_location="cpu", weights_only=True)
    raw_opt = optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer
    raw_opt.load_state_dict(opt_sd)
    if accelerator.is_main_process:
        print(f"[Resume] Loaded optimizer state ({len(opt_sd['state'])} param states)")

    # 3. LR scheduler state
    sched_path = os.path.join(ckpt_dir, "scheduler.pt")
    sched_sd = torch.load(sched_path, map_location="cpu", weights_only=True)
    raw_sched = lr_scheduler.scheduler if hasattr(lr_scheduler, 'scheduler') else lr_scheduler
    raw_sched.load_state_dict(sched_sd)

    # 4. RNG states (each rank loads its own)
    rng_path = os.path.join(ckpt_dir, f"random_states_{accelerator.process_index}.pkl")
    if os.path.isfile(rng_path):
        with open(rng_path, "rb") as f:
            rng_state = pickle.load(f)
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.random.set_rng_state(rng_state["torch_cpu"])
        if "torch_cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["torch_cuda"], accelerator.device)
        if accelerator.is_main_process:
            print(f"[Resume] Restored RNG states for {accelerator.num_processes} processes")
    else:
        if accelerator.is_main_process:
            print(f"[Resume] WARNING: No RNG state at {rng_path}")

    accelerator.wait_for_everyone()


# ============================================================================
# Training Loop
# ============================================================================

def launch_training(dataset, model, args):
    # Extend NCCL timeout to 2 hours for long validation runs
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[pg_kwargs],
    )

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        return batch[0] if batch else None

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler,
    )

    # --- Post-prepare setup ---
    unwrapped_init = accelerator.unwrap_model(model)

    # Re-apply training modes (accelerator.prepare may alter training state)
    unwrapped_init._set_training_modes()
    unwrapped_init.verbose_vram = getattr(args, "verbose_vram", False)

    # Optional: offload frozen encoders to CPU (T5 + Wav2Vec only).
    # VAE stays on GPU — it's only 0.26 GB and backward needs it for aux loss gradients.
    if getattr(args, "offload_frozen", False):
        unwrapped_init.offload_frozen = True
        unwrapped_init.text_encoder.to("cpu")
        unwrapped_init.wav2vec.to("cpu")
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        alloc = torch.cuda.memory_allocated() / 1e9
        accelerator.print(f"[VRAM] After prepare: {alloc:.2f} GB on GPU, "
                          f"dit.training={unwrapped_init.dit.training}, "
                          f"offload_frozen={unwrapped_init.offload_frozen}")

    # Condition dropout
    unwrapped_init.text_drop_prob = args.text_drop_prob
    unwrapped_init.audio_drop_prob = args.audio_drop_prob
    if args.text_drop_prob > 0 or args.audio_drop_prob > 0:
        accelerator.print(
            f"[Train] Condition dropout: text={args.text_drop_prob}, audio={args.audio_drop_prob}"
        )

    if unwrapped_init.syncnet is not None:
        unwrapped_init.syncnet = unwrapped_init.syncnet.to(
            device=accelerator.device, dtype=torch.float16
        )
    if unwrapped_init.lpips_func is not None:
        unwrapped_init.lpips_func = unwrapped_init.lpips_func.to(device=accelerator.device)
    if unwrapped_init.trepa_func is not None:
        unwrapped_init.trepa_func.model = unwrapped_init.trepa_func.model.to(
            device=accelerator.device
        )

    # --- Preload V2V validation samples (recon + mixed) ---
    val_recon_samples = []
    val_mixed_samples = []
    if accelerator.is_main_process:
        if args.val_data_recon:
            val_recon_samples = load_v2v_validation_data(
                args.val_data_recon, max_samples=args.val_max_samples
            )
        if args.val_data_mixed:
            val_mixed_samples = load_v2v_validation_data(
                args.val_data_mixed, max_samples=args.val_max_samples
            )

    has_val_samples = len(val_recon_samples) > 0 or len(val_mixed_samples) > 0

    # --- SyncNet ---
    syncnet, syncnet_detector = None, None
    if accelerator.is_main_process:
        syncnet, syncnet_detector = load_syncnet(args)

    # --- Loss tracking ---
    # Note: ema_loss and accumulators are NOT restored on resume (same as StableAvatar).
    # Wandb dashboard will show a discontinuity at the resume point.
    ema_loss = None
    ema_beta = 0.99
    window_loss_sum = window_loss_count = 0.0
    cum_loss_sum = cum_loss_count = 0.0
    ga_loss_sum = ga_loss_count = 0.0
    global_step = 0
    first_epoch = 0

    # --- Resume from checkpoint ---
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            ckpt_path = args.resume_from_checkpoint
        else:
            # Find the most recent checkpoint-N directory
            dirs = [
                d for d in os.listdir(args.output_path)
                if d.startswith("checkpoint-")
            ] if os.path.isdir(args.output_path) else []
            if dirs:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                ckpt_path = os.path.join(args.output_path, dirs[-1])
            else:
                ckpt_path = None

        if ckpt_path is None or not os.path.isdir(ckpt_path):
            accelerator.print(
                f"[Resume] Checkpoint '{args.resume_from_checkpoint}' not found. Starting fresh."
            )
        else:
            accelerator.print(f"[Resume] Loading state from {ckpt_path}")
            # Detect format: lean (trainable_weights.pt) vs legacy (model.safetensors)
            if os.path.isfile(os.path.join(ckpt_path, "trainable_weights.pt")):
                load_training_state(
                    accelerator, model, optimizer, lr_scheduler, ckpt_path,
                )
            else:
                accelerator.print("[Resume] Legacy checkpoint — using accelerator.load_state")
                accelerator.load_state(ckpt_path)
            global_step = int(os.path.basename(ckpt_path).split("-")[1])
            num_update_steps_per_epoch = math.ceil(
                len(dataloader) / args.gradient_accumulation_steps
            )
            first_epoch = global_step // num_update_steps_per_epoch
            accelerator.print(
                f"[Resume] Resuming from step {global_step}, epoch {first_epoch}"
            )

    # --- wandb ---
    use_wandb = getattr(args, "use_wandb", False) and accelerator.is_main_process
    if use_wandb:
        try:
            import wandb
            api_key = getattr(args, "wandb_api_key", None)
            if api_key:
                wandb.login(key=api_key, relogin=True)
            # Determine run ID: explicit arg > saved in checkpoint > None (new run)
            # Use --wandb_fresh_run to force a new run even when resuming training
            wandb_run_id = getattr(args, "wandb_run_id", None)
            if wandb_run_id is None and global_step > 0 and not getattr(args, "wandb_fresh_run", False):
                # Resuming from checkpoint — try to load saved wandb run ID
                wandb_id_path = os.path.join(
                    args.output_path, f"checkpoint-{global_step}", "wandb_id.txt"
                )
                if os.path.isfile(wandb_id_path):
                    with open(wandb_id_path) as f:
                        wandb_run_id = f.read().strip()
                    print(f"[W&B] Loaded saved run ID from checkpoint: {wandb_run_id}")
            wandb_config = {k: v for k, v in vars(args).items() if k != "wandb_api_key"}
            wandb.init(
                project=getattr(args, "wandb_project", "OmniAvatar-V2V"),
                entity=getattr(args, "wandb_entity", None),
                name=getattr(args, "wandb_run_name", None) or os.path.basename(args.output_path),
                tags=args.wandb_tags.split(",") if getattr(args, "wandb_tags", None) else None,
                config=wandb_config,
                id=wandb_run_id,
                resume="allow" if wandb_run_id else None,
                dir=args.output_path,
            )
            print(f"[W&B] Initialized: {wandb.run.id}")
        except Exception as e:
            print(f"[W&B] Init failed: {e}")
            use_wandb = False

    # --- Validation at step 0 ---
    if getattr(args, "validate_at_start", False) and accelerator.is_main_process and has_val_samples:
        print("[Validation] Running validation at step 0 (baseline)...")
        unwrapped = accelerator.unwrap_model(model)
        val_device = next(unwrapped.dit.parameters()).device
        if val_recon_samples:
            run_v2v_validation_loop(
                unwrapped, val_recon_samples, "recon", args, global_step, val_device,
                syncnet, syncnet_detector, use_wandb,
            )
        if val_mixed_samples:
            run_v2v_validation_loop(
                unwrapped, val_mixed_samples, "mixed", args, global_step, val_device,
                syncnet, syncnet_detector, use_wandb,
            )

    # --- Training loop ---
    consecutive_nones = 0
    # Number of batches to skip in the first resumed epoch
    resume_batches_to_skip = 0
    if global_step > 0 and first_epoch < args.num_epochs:
        resume_batches_to_skip = (global_step - first_epoch * math.ceil(
            len(dataloader) / args.gradient_accumulation_steps
        )) * args.gradient_accumulation_steps
        accelerator.print(f"[Resume] Skipping {resume_batches_to_skip} batches in epoch {first_epoch}")

    for epoch_id in range(first_epoch, args.num_epochs):
        # Skip already-processed batches when resuming mid-epoch
        if epoch_id == first_epoch and resume_batches_to_skip > 0:
            active_dataloader = accelerator.skip_first_batches(dataloader, resume_batches_to_skip)
            resume_batches_to_skip = 0  # only skip once
        else:
            active_dataloader = dataloader

        epoch_pbar = tqdm(
            active_dataloader, desc=f"Epoch {epoch_id + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for data in epoch_pbar:
            if data is None:
                consecutive_nones += 1
                if consecutive_nones >= 50:
                    raise RuntimeError(
                        f"50 consecutive None samples — check dataset path and data files"
                    )
                continue
            consecutive_nones = 0

            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)
                loss = model(data)
                accelerator.backward(loss)
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    ga_loss_sum += loss.detach()
                    ga_loss_count += 1

                if accelerator.sync_gradients:
                    step_loss = (ga_loss_sum / max(1, ga_loss_count)).item()
                    ga_loss_sum = ga_loss_count = 0.0
                    global_step += 1

                    cum_loss_sum += step_loss
                    cum_loss_count += 1
                    window_loss_sum += step_loss
                    window_loss_count += 1
                    ema_loss = step_loss if ema_loss is None else (
                        ema_beta * ema_loss + (1 - ema_beta) * step_loss
                    )

                    epoch_pbar.set_postfix(
                        loss=f"{step_loss:.4f}", ema=f"{ema_loss:.4f}", step=global_step,
                    )

                    if use_wandb and global_step % args.wandb_log_every == 0:
                        try:
                            import wandb
                            log_dict = {
                                "loss/step": step_loss, "loss/ema": ema_loss,
                                "loss/window_mean": window_loss_sum / max(1, window_loss_count),
                                "loss/cum_mean": cum_loss_sum / max(1, cum_loss_count),
                                "train/epoch": epoch_id,
                                "train/lr": lr_scheduler.get_last_lr()[0],
                            }
                            uw = accelerator.unwrap_model(model)
                            if uw._last_aux_losses:
                                for k, v in uw._last_aux_losses.items():
                                    log_dict[f"aux/{k}"] = v.item() if hasattr(v, 'item') else v
                            wandb.log(log_dict, step=global_step)
                        except Exception as e:
                            print(f"[W&B] log error: {e}")
                        window_loss_sum = window_loss_count = 0.0

                    if args.save_steps and global_step % args.save_steps == 0:
                        # Lean training state — all ranks must participate (collective op)
                        ckpt_dir = os.path.join(args.output_path, f"checkpoint-{global_step}")
                        save_training_state(
                            accelerator, model, optimizer, lr_scheduler, ckpt_dir, args,
                        )
                        if accelerator.is_main_process:
                            # Save wandb run ID for seamless resume
                            if use_wandb:
                                import wandb
                                with open(os.path.join(ckpt_dir, "wandb_id.txt"), "w") as f:
                                    f.write(wandb.run.id)
                            print(f"[Checkpoint] Saved lean training state to {ckpt_dir}")
                            # Rotate old checkpoints (save-then-rotate for crash safety)
                            if args.checkpoints_total_limit is not None:
                                checkpoints = [
                                    d for d in os.listdir(args.output_path)
                                    if d.startswith("checkpoint-") and d != f"checkpoint-{global_step}"
                                ]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                                while len(checkpoints) >= args.checkpoints_total_limit:
                                    old_ckpt = checkpoints.pop(0)
                                    shutil.rmtree(os.path.join(args.output_path, old_ckpt))
                                    print(f"[Checkpoint] Removed old checkpoint: {old_ckpt}")
                        # LoRA-only weights (for inference)
                        save_checkpoint(accelerator, model, args.output_path, global_step, args)

                    lr_scheduler.step()

                    # Validation
                    if (
                        args.validation_steps > 0
                        and global_step % args.validation_steps == 0
                        and accelerator.is_main_process
                        and has_val_samples
                    ):
                        print(f"\n[Validation] Running at step {global_step}...")
                        torch.cuda.empty_cache()
                        unwrapped = accelerator.unwrap_model(model)
                        val_device = next(unwrapped.dit.parameters()).device
                        if val_recon_samples:
                            run_v2v_validation_loop(
                                unwrapped, val_recon_samples, "recon", args,
                                global_step, val_device,
                                syncnet, syncnet_detector, use_wandb,
                            )
                        if val_mixed_samples:
                            run_v2v_validation_loop(
                                unwrapped, val_mixed_samples, "mixed", args,
                                global_step, val_device,
                                syncnet, syncnet_detector, use_wandb,
                            )

    # Final checkpoint — all ranks must participate (collective op)
    ckpt_dir = os.path.join(args.output_path, f"checkpoint-{global_step}")
    if not os.path.isdir(ckpt_dir):  # skip if periodic save already created it
        save_training_state(
            accelerator, model, optimizer, lr_scheduler, ckpt_dir, args,
        )
        if accelerator.is_main_process:
            if use_wandb:
                import wandb
                with open(os.path.join(ckpt_dir, "wandb_id.txt"), "w") as f:
                    f.write(wandb.run.id)
            print(f"[Checkpoint] Saved final lean training state to {ckpt_dir}")
    save_checkpoint(accelerator, model, args.output_path, f"final-{global_step}", args)

    if use_wandb:
        try:
            import wandb
            wandb.summary["final/loss_ema"] = ema_loss
            wandb.summary["final/loss_cum_mean"] = cum_loss_sum / max(1, cum_loss_count)
            wandb.summary["final/global_step"] = global_step
            wandb.finish()
        except Exception:
            pass


# ============================================================================
# Argument Parser
# ============================================================================

def train_parser():
    parser = argparse.ArgumentParser(description="OmniAvatar V2V training script.")

    # Model paths
    parser.add_argument("--dit_paths", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--wav2vec_path", type=str, required=True)
    parser.add_argument("--omniavatar_ckpt", type=str, default=None,
                        help="Path to OmniAvatar checkpoint (33ch) for finetuning. Omit for from-scratch.")

    # Dataset — V2V specific
    parser.add_argument("--data_list_path", type=str, required=True,
                        help="Text file with one video directory per line")
    parser.add_argument("--latentsync_mask_path", type=str, required=True,
                        help="Path to LatentSync spatial mask PNG")
    parser.add_argument("--use_precomputed_vae", action="store_true",
                        help="Load precomputed VAE latents from vae_latents.pt")
    parser.add_argument("--use_precomputed_audio", action="store_true",
                        help="Load precomputed audio from audio_emb_omniavatar.pt")
    parser.add_argument("--use_precomputed_text_emb", action="store_true",
                        help="Load precomputed T5 text embeddings from text_emb.pt")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=25)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=float, default=64.0)
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true", default=False)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Condition dropout
    parser.add_argument("--text_drop_prob", type=float, default=0.1)
    parser.add_argument("--audio_drop_prob", type=float, default=0.1)

    # VRAM management
    parser.add_argument("--offload_frozen", action="store_true", default=False,
                        help="Offload frozen encoders (T5, VAE, Wav2Vec) to CPU between uses. "
                             "Saves ~12 GB VRAM but adds latency per step.")
    parser.add_argument("--verbose_vram", action="store_true", default=False,
                        help="Print VRAM usage at each stage of the forward pass.")

    # Checkpoint
    parser.add_argument("--output_path", type=str, default="./checkpoints/omniavatar-v2v-14b")
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint dir (lean or legacy format), or 'latest' to auto-detect.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None,
                        help="Max number of training checkpoints to keep. Oldest removed first.")

    # Wandb
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="OmniAvatar-V2V")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--wandb_log_every", type=int, default=10)
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Wandb run ID to resume logging into an existing run.")
    parser.add_argument("--wandb_fresh_run", action="store_true", default=False,
                        help="Force a new wandb run even when resuming from checkpoint.")

    # Validation — V2V specific (recon + mixed)
    parser.add_argument("--val_data_recon", type=str, default=None,
                        help="Directory with reconstruction validation data")
    parser.add_argument("--val_data_mixed", type=str, default=None,
                        help="Directory with mixed (cross-identity) validation data")
    parser.add_argument("--val_max_samples", type=int, default=4)
    parser.add_argument("--validation_steps", type=int, default=0)
    parser.add_argument("--val_num_inference_steps", type=int, default=25)
    parser.add_argument("--val_cfg_scale", type=float, default=4.5)
    parser.add_argument("--validate_at_start", action="store_true", default=False)

    # SyncNet (validation metrics)
    parser.add_argument("--compute_sync_metrics", action="store_true", default=False)
    parser.add_argument("--syncnet_model_path", type=str,
                        default="/home/work/.local/LatentSync/checkpoints/auxiliary/syncnet_v2.model")
    parser.add_argument("--s3fd_model_path", type=str,
                        default="/home/work/.local/LatentSync/checkpoints/auxiliary/sfd_face.pth")

    # Auxiliary losses
    parser.add_argument("--use_sync_loss", action="store_true", default=False)
    parser.add_argument("--use_lpips_loss", action="store_true", default=False)
    parser.add_argument("--use_trepa_loss", action="store_true", default=False)
    parser.add_argument("--aux_recon_weight", type=float, default=1.0)
    parser.add_argument("--aux_sync_weight", type=float, default=0.1)
    parser.add_argument("--aux_lpips_weight", type=float, default=0.1)
    parser.add_argument("--aux_trepa_weight", type=float, default=10.0)
    parser.add_argument("--aux_num_frames", type=int, default=21)
    parser.add_argument("--sync_chunk_size", type=int, default=16)
    parser.add_argument("--sync_chunk_stride", type=int, default=8)
    parser.add_argument("--sync_num_supervised_frames", type=int, default=80)
    parser.add_argument("--use_vae_gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--syncnet_config_path", type=str,
                        default="/home/work/.local/Self-Forcing_LipSync_StableAvatar/latentsync_models/configs/syncnet/syncnet_16_pixel_attn.yaml")
    parser.add_argument("--syncnet_checkpoint_path", type=str,
                        default="/home/work/.local/Self-Forcing_LipSync_StableAvatar/examples/wanvideo/model_training/checkpoints/stable_syncnet.pt")
    parser.add_argument("--trepa_checkpoint_path", type=str,
                        default="/home/work/.local/Self-Forcing_LipSync_StableAvatar/examples/wanvideo/model_training/checkpoints/auxiliary/vit_g_hybrid_pt_1200e_ssv2_ft.pth")

    return parser


# ============================================================================
# Main
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config_yaml(args):
    if not args.config:
        return args
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
        elif isinstance(getattr(args, key), bool) and not getattr(args, key) and isinstance(value, bool):
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    parser = train_parser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config. CLI args override config values.")
    args = parser.parse_args()
    args = load_config_yaml(args)

    set_seed(args.seed)

    # Load LatentSync mask
    latentsync_mask = None
    if args.latentsync_mask_path and os.path.exists(args.latentsync_mask_path):
        mask_img = Image.open(args.latentsync_mask_path)
        mask_arr = np.array(mask_img)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        latentsync_mask = torch.from_numpy(mask_arr.astype(np.float32) / 255.0)
        print(f"[Mask] Loaded LatentSync mask: {latentsync_mask.shape}")

    # Dataset
    dataset = OmniAvatarV2VDataset(
        data_list_path=args.data_list_path,
        num_frames=args.num_frames,
        height=args.height, width=args.width,
        sample_rate=args.sample_rate, fps=args.fps,
        repeat=args.dataset_repeat,
        latentsync_mask_path=args.latentsync_mask_path,
        use_precomputed_vae=args.use_precomputed_vae,
        use_precomputed_audio=args.use_precomputed_audio,
        use_precomputed_text_emb=args.use_precomputed_text_emb,
    )
    print(f"[Dataset] {len(dataset)} samples (precomputed_vae={args.use_precomputed_vae}, precomputed_audio={args.use_precomputed_audio}, precomputed_text={args.use_precomputed_text_emb})")

    # Model
    model = OmniAvatarV2VTrainingModule(
        dit_paths=args.dit_paths,
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        wav2vec_path=args.wav2vec_path,
        omniavatar_ckpt=args.omniavatar_ckpt,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        latentsync_mask=latentsync_mask,
        num_training_frames=args.num_frames,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total: {total:,} | Trainable: {trainable:,} ({trainable/total*100:.2f}%)")

    # Setup auxiliary losses
    if args.use_sync_loss or args.use_lpips_loss or args.use_trepa_loss:
        model.setup_aux_losses(args)

    # Train
    launch_training(dataset, model, args)
