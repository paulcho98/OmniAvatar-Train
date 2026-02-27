"""OmniAvatar 14B Training Script.

Reconstructs the training pipeline for OmniAvatar by using the inference DiT model
directly (guaranteeing forward-pass parity) with HuggingFace Accelerate for distributed
training, LoRA for parameter-efficient finetuning, and wandb for logging.

Usage:
    # Finetune from OmniAvatar checkpoint, 2 GPUs
    CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file configs/accelerate_2gpus.yaml \\
        scripts/train.py --dit_paths "..." --text_encoder_path "..." --vae_path "..." \\
        --wav2vec_path "..." --omniavatar_ckpt "pretrained_models/OmniAvatar-14B/pytorch_model.pt" \\
        --dataset_base_path "/path/to/videos" --dataset_metadata_path "metadata.csv" \\
        --use_gradient_checkpointing --use_wandb --validate_at_start

    # From scratch (omit --omniavatar_ckpt)
    CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 scripts/train.py ...
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
import pandas as pd
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
# The DiT model reads args.use_audio, args.sp_size, args.model_config at
# construction time (wan_video_dit.py:307-318).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import OmniAvatar.utils.args_config as args_module


def setup_omniavatar_args():
    """Configure the global args singleton that OmniAvatar's DiT depends on."""
    args_obj = argparse.Namespace(
        use_audio=True,
        sp_size=1,           # No sequence parallel — Accelerate handles DDP
        model_config={"in_dim": 33, "audio_hidden_size": 32},
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

class OmniAvatarTrainingDataset(torch.utils.data.Dataset):
    """Dataset for OmniAvatar training.

    CSV metadata with columns: video, prompt
    Videos are in base_path directory.

    For each sample:
    - Loads video frames (PIL Images)
    - Extracts audio from video (# To be later removed)
    - First frame is used as reference image
    """

    def __init__(self, base_path, metadata_path, num_frames=81, height=512,
                 width=512, sample_rate=16000, fps=25, repeat=1):
        self.base_path = base_path
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.sample_rate = sample_rate
        self.fps = fps
        self.repeat = repeat

        metadata = pd.read_csv(metadata_path)
        self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

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
        # Enforce Wan 2.1 temporal constraint: num_frames % 4 == 1
        while num_frames > 1 and num_frames % 4 != 1:
            num_frames -= 1
        frames = []
        for i in range(num_frames):
            frame = Image.fromarray(reader.get_data(i))
            frame = self._crop_and_resize(frame, self.height, self.width)
            frames.append(frame)
        reader.close()
        return frames

    def _extract_audio(self, video_path):
        """Extract audio from video file. # To be later removed"""
        try:
            audio, _ = librosa.load(video_path, sr=self.sample_rate)
            return audio
        except Exception:
            pass
        # Fallback: ffmpeg extraction
        tmp_path = os.path.join(tempfile.gettempdir(), f"_audio_{os.getpid()}.wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", str(self.sample_rate), "-ac", "1", tmp_path],
                capture_output=True, check=True,
            )
            audio, _ = librosa.load(tmp_path, sr=self.sample_rate)
            return audio
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def __getitem__(self, data_id):
        row = self.data[data_id % len(self.data)].copy()
        video_path = os.path.join(self.base_path, row["video"])
        try:
            frames = self._load_video(video_path)
            if len(frames) < 5:
                return None
            audio = self._extract_audio(video_path)
            # Trim audio to match video duration (critical for alignment)
            num_frames = len(frames)
            target_samples = int(num_frames / self.fps * self.sample_rate)
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            return {
                "video": frames,
                "prompt": row.get("prompt", "a person is talking"),
                "audio": audio,
                "ref_image": frames[0],
                "video_path": video_path,
            }
        except Exception as e:
            print(f"[Dataset] Error loading {video_path}: {e}")
            return None

    def __len__(self):
        return len(self.data) * self.repeat


# ============================================================================
# Training Module
# ============================================================================

class OmniAvatarTrainingModule(nn.Module):
    """Training module for OmniAvatar 14B.

    Uses OmniAvatar's own DiT model (not DiffSynth's) for guaranteed
    forward-pass parity with inference.

    Frozen: T5 text encoder, VAE, Wav2Vec2
    Trainable (LoRA): DiT attention (q,k,v,o) and FFN (ffn.0, ffn.2)
    Trainable (full): AudioPack, audio_cond_projs, patch_embedding
    """

    def __init__(self, dit_paths, text_encoder_path, vae_path, wav2vec_path,
                 omniavatar_ckpt=None, lora_rank=128, lora_alpha=64,
                 lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                 use_gradient_checkpointing=True,
                 use_gradient_checkpointing_offload=False,
                 dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        # --- 1. Load base models via ModelManager ---
        # infer=False triggers xavier_uniform_(gain=0.05) init for new params,
        # then smart_load_weights overlays base weights (model_manager.py:27-33)
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
            init_lora_weights=True,  # "kaiming" maps to True in PEFT
            target_modules=lora_target_modules.split(","),
        )
        self.pipe.dit = inject_adapter_in_model(lora_config, self.pipe.dit)
        # Upcast LoRA params
        for param in self.pipe.dit.parameters():
            if param.requires_grad:
                param.data = param.to(dtype)

        # --- 6. Load OmniAvatar checkpoint (finetuning) ---
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
        )  # stays float32 — matches inference.py:97
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

        # Condition dropout (set from args in launch_training)
        self.text_drop_prob = 0.0
        self.audio_drop_prob = 0.0
        self._empty_text_cache = None

        # Auxiliary losses (initialized via setup_aux_losses if enabled)
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
        """Set train/eval modes explicitly following DiffSynth's freeze_except pattern.

        Critical: wan_video_dit.py:405 checks `self.training` to enable gradient
        checkpointing. Without explicit .train(), this can silently be False after
        accelerator.prepare(), causing 105+ GB activation memory instead of ~17 GB.
        """
        self.pipe.dit.train()
        self.pipe.text_encoder.eval()
        self.pipe.vae.eval()
        self.wav2vec.eval()
        if self.syncnet is not None:
            self.syncnet.eval()
        if self.lpips_func is not None:
            self.lpips_func.eval()
        if self.trepa_func is not None:
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
        """Return trainable parameters for optimizer."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_names(self):
        return {n for n, p in self.named_parameters() if p.requires_grad}

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        """Export only trainable parameters."""
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

        # Pad to match video length
        samples_per_frame = self.wav_feature_extractor.sampling_rate // 25  # 640
        target_samples = num_video_frames * samples_per_frame
        if input_values.shape[1] < target_samples:
            input_values = F.pad(input_values, (0, target_samples - input_values.shape[1]))

        hidden_states = self.wav2vec(
            input_values, seq_len=num_video_frames, output_hidden_states=True,
        )
        # Concatenate: last_hidden_state + 13 hidden_states = 14 × 768 = 10752
        audio_emb = hidden_states.last_hidden_state
        for hs in hidden_states.hidden_states:
            audio_emb = torch.cat((audio_emb, hs), -1)
        return audio_emb  # (1, num_video_frames, 10752)

    @torch.no_grad()
    def encode_text(self, prompt, device):
        """Encode text prompt via T5."""
        self.text_encoder.to(device)
        prompt_emb = self.prompter.encode_prompt(prompt, positive=True, device=device)
        return prompt_emb  # (1, seq_len, 4096)

    @torch.no_grad()
    def encode_video(self, video_frames, device):
        """Encode video frames via VAE → latent."""
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

    @staticmethod
    def prepare_reference_input(ref_latent, num_lat_frames):
        """Prepare y tensor: ref latent repeated + binary mask. Matches inference.py:271-275."""
        image_cat = ref_latent.repeat(1, 1, num_lat_frames, 1, 1)
        msk = torch.zeros_like(image_cat[:, :1])
        msk[:, :, 1:] = 1
        return torch.cat([image_cat, msk], dim=1)  # (1, 17, T, H, W)

    # ------------------------------------------------------------------
    # Auxiliary losses setup and computation
    # ------------------------------------------------------------------

    def setup_aux_losses(self, args):
        """Load frozen auxiliary models for training losses (SyncNet, LPIPS, TREPA)."""
        self.aux_recon_weight = args.aux_recon_weight
        self.aux_sync_weight = args.aux_sync_weight
        self.aux_lpips_weight = args.aux_lpips_weight
        self.aux_trepa_weight = args.aux_trepa_weight
        self.aux_num_frames = args.aux_num_frames
        self.sync_chunk_size = args.sync_chunk_size
        self.sync_chunk_stride = args.sync_chunk_stride
        self.sync_num_supervised_frames = args.sync_num_supervised_frames

        # Enable VAE decoder gradient checkpointing
        if args.use_vae_gradient_checkpointing:
            self.pipe.vae.model.decoder.gradient_checkpointing = True
            print("[AuxLoss] Enabled VAE decoder gradient checkpointing")

        latentsync_path = "/home/work/.local/Self-Forcing_LipSync_StableAvatar"

        # SyncNet
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
            # Import melspectrogram for sync loss
            from latentsync_models.audio import melspectrogram as _mel_fn
            self._melspectrogram = _mel_fn
            print(f"[AuxLoss] Loaded SyncNet from {args.syncnet_checkpoint_path}")

        # LPIPS
        if args.use_lpips_loss:
            import lpips as lpips_pkg
            self.lpips_func = lpips_pkg.LPIPS(net="vgg")
            self.lpips_func.requires_grad_(False)
            self.lpips_func.eval()
            print("[AuxLoss] Loaded LPIPS (VGG)")

        # TREPA
        if args.use_trepa_loss:
            sys.path.insert(0, latentsync_path)
            from latentsync_models.trepa.loss import TREPALoss
            self.trepa_func = TREPALoss(
                device="cpu", ckpt_path=args.trepa_checkpoint_path, with_cp=True
            )
            print(f"[AuxLoss] Loaded TREPA from {args.trepa_checkpoint_path}")

    def _decode_for_aux_loss(self, latents, device, with_grad=True):
        """Decode latents to RGB via VAE, bypassing CPU offload in WanVideoVAE.decode().

        Uses self.pipe.vae.model.decode() directly with non-inplace clamp for autograd.
        Gradient checkpointing (if enabled) is handled internally by Decoder3d.
        """
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
        """Chunked SyncNet loss. pred_rgb: [B,3,T,H,W] in [-1,1], audio_np: numpy 16kHz."""
        from einops import rearrange

        B, C, T, H, W = pred_rgb.shape
        chunk_size = self.sync_chunk_size
        stride = self.sync_chunk_stride
        num_supervised = self.sync_num_supervised_frames
        fps = 25
        sr = 16000

        # Flatten and resize to SyncNet resolution (256x256)
        pred_flat = rearrange(pred_rgb, "b c t h w -> (b t) c h w")
        pred_resized = F.interpolate(
            pred_flat, size=(256, 256), mode="bicubic", align_corners=False
        )
        pred_lower = pred_resized[:, :, 128:, :]  # Lower half

        chunk_losses = []
        num_chunks = max(1, math.ceil((min(T, num_supervised) - chunk_size) / stride) + 1)

        for i in range(num_chunks):
            start = i * stride
            end = min(start + chunk_size, T)
            if end - start < chunk_size:
                continue  # SyncNet expects exactly chunk_size frames

            try:
                # Visual: [B, chunk*3, 128, 256]
                chunk_frames = pred_lower[start:end]
                vis_input = rearrange(chunk_frames, "(b t) c h w -> b (t c) h w", b=B)

                # Audio: extract mel for this chunk's time window
                start_sample = int(start / fps * sr)
                end_sample = int(end / fps * sr)
                audio_seg = audio_np[start_sample:end_sample]
                if len(audio_seg) < 100:
                    audio_seg = np.zeros(int((end - start) / fps * sr), dtype=np.float32)
                mel = self._melspectrogram(audio_seg)  # [80, time_steps]
                mel_window = math.ceil((end - start) / 5.0 * 16)
                mel_t = torch.from_numpy(mel).float()
                mel_t = F.pad(mel_t, (0, max(0, mel_window - mel_t.shape[1])))[:, :mel_window]
                mel_t = mel_t.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float16)

                # SyncNet forward
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
        """LPIPS on lower half of face. pred/gt_rgb: [B,3,T,H,W] in [-1,1]."""
        from einops import rearrange

        pred_flat = rearrange(pred_rgb, "b c t h w -> (b t) c h w")
        gt_flat = rearrange(gt_rgb, "b c t h w -> (b t) c h w")
        H = pred_flat.shape[2]
        return self.lpips_func(
            pred_flat[:, :, H // 2 :, :].float(),
            gt_flat[:, :, H // 2 :, :].float(),
        ).mean()

    def _compute_trepa_loss(self, pred_rgb, gt_rgb):
        """TREPA temporal consistency loss. pred/gt_rgb: [B,3,T,H,W] in [-1,1]."""
        return self.trepa_func(pred_rgb, gt_rgb)

    # ------------------------------------------------------------------
    # Forward pass — flow matching training loss
    # ------------------------------------------------------------------

    def forward(self, data):
        """Training forward pass with condition dropout and optional auxiliary losses."""
        device = next(self.dit.parameters()).device

        # Defensive: ensure DiT is in training mode for gradient checkpointing
        if not self.dit.training:
            self.dit.train()
        self._log_vram("forward-start")

        # 1. Encode text
        if self.offload_frozen:
            self.text_encoder.to(device)
        context = self.encode_text(data["prompt"], device)
        if self.offload_frozen:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        self._log_vram("after-text-encode")

        # 2. Encode audio
        if self.offload_frozen:
            self.wav2vec.to(device)
        num_video_frames = len(data["video"])
        audio_emb = self.encode_audio(data["audio"], num_video_frames, device)
        audio_emb = audio_emb.to(dtype=self.dtype, device=device)
        if self.offload_frozen:
            self.wav2vec.to("cpu")
            torch.cuda.empty_cache()
        self._log_vram("after-audio-encode")

        # 3. Condition dropout for CFG training
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

        # 4. Encode GT video (VAE always on GPU — only 0.26 GB)
        input_latents = self.encode_video(data["video"], device)
        self._log_vram("after-vae-encode")

        # 5. Prepare reference frame
        ref_latent = input_latents[:, :, :1]
        T_lat = input_latents.shape[2]
        y = self.prepare_reference_input(ref_latent, T_lat)

        # 6. Sample noise + timestep
        noise = torch.randn_like(input_latents)
        num_timesteps = len(self.scheduler.timesteps)
        timestep_id = torch.randint(0, num_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=device)

        noisy_latents = self.scheduler.add_noise(input_latents, noise, timestep)
        training_target = self.scheduler.training_target(input_latents, noise, timestep)
        self._log_vram("before-dit-forward")

        # 7. DiT forward (THE critical call)
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

        # 9. Auxiliary losses on decoded x_0 (SyncNet, LPIPS, TREPA)
        use_any_aux = (self.syncnet is not None or self.lpips_func is not None
                       or self.trepa_func is not None)
        if use_any_aux:
            self._log_vram("before-aux-losses")
            mse_loss_val = loss.detach()  # Save before combining

            sigma = self.scheduler.sigmas[timestep_id].to(
                device=device, dtype=noise_pred.dtype
            )
            sync_len = self.aux_num_frames if self.aux_num_frames > 0 else T_lat

            # x_0 prediction: x_0 = x_t - sigma * noise_pred (flow matching)
            x_0_pred = noisy_latents[:, :, :sync_len] - sigma * noise_pred[:, :, :sync_len]
            x_0_pred = torch.clamp(x_0_pred, -10, 10)

            # VAE always on GPU (0.26 GB) — backward needs it for gradient flow
            pred_rgb = self._decode_for_aux_loss(x_0_pred, device, with_grad=True)
            self._log_vram("after-vae-decode-pred")

            gt_rgb = self._decode_for_aux_loss(
                input_latents[:, :, :sync_len].detach(), device, with_grad=False
            )
            self._log_vram("after-vae-decode-gt")

            # Compute enabled losses
            sync_loss = torch.tensor(0.0, device=device)
            lpips_loss = torch.tensor(0.0, device=device)
            trepa_loss = torch.tensor(0.0, device=device)

            if self.syncnet is not None:
                sync_loss = self._compute_sync_loss(pred_rgb, data["audio"], device)
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
            }

        return loss


# ============================================================================
# Validation
# ============================================================================

@torch.no_grad()
def run_validation(model, val_sample, device, num_inference_steps=25,
                   guidance_scale=4.5, negative_prompt=""):
    """Generate video from pure noise with ref frame + audio + CFG.

    Matches inference.py behavior:
    - Reference image VAE-encoded as single frame (not extracted from full video)
    - Fixed-frame overwriting each denoising step (anchors reference frame)
    - Text+audio CFG with negative prompt
    """
    model.eval()
    pipe = model.pipe

    num_video_frames = len(val_sample["video"])

    # 1. Encode text (transient GPU load if offloading)
    if model.offload_frozen:
        model.text_encoder.to(device)
    context = model.encode_text(val_sample["prompt"], device)

    # 2. Encode audio
    if model.offload_frozen:
        model.wav2vec.to(device)
    audio_emb = model.encode_audio(val_sample["audio"], num_video_frames, device)
    audio_emb = audio_emb.to(dtype=model.dtype, device=device)
    if model.offload_frozen:
        model.wav2vec.to("cpu")
        torch.cuda.empty_cache()

    # Negative text for CFG
    neg_context = model.encode_text(negative_prompt, device) if guidance_scale != 1.0 else None
    if model.offload_frozen:
        model.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    # 3. Encode reference image as SINGLE frame (matching inference.py:270)
    ref_image = val_sample["ref_image"]
    ref_np = np.array(ref_image, dtype=np.float32) * (2.0 / 255.0) - 1.0
    ref_t = torch.from_numpy(ref_np).permute(2, 0, 1)  # (C, H, W)
    ref_video = ref_t.unsqueeze(1).unsqueeze(0)  # (1, C, 1, H, W)
    # VAE stays on GPU (only 0.26 GB)
    old_device = pipe.device
    pipe.device = device
    ref_latent = model.vae.encode(
        ref_video.to(dtype=model.dtype, device=device),
        device=device, tiled=True, tile_size=(34, 34), tile_stride=(18, 16),
    ).to(dtype=model.dtype, device=device)  # (1, 16, 1, H_lat, W_lat)

    # 4. Compute T_lat from video frame count
    T_lat = (num_video_frames + 3) // 4

    # 5. Build reference input y = [ref_latent repeated + mask] (matching inference.py:272-275)
    y = model.prepare_reference_input(ref_latent, T_lat)

    # 6. Build the "clean" latent for fixed-frame overwriting (matching inference.py:306)
    lat = torch.cat([
        ref_latent,
        torch.zeros(1, 16, T_lat - 1, ref_latent.shape[3], ref_latent.shape[4],
                     dtype=model.dtype, device=device),
    ], dim=2)  # (1, 16, T_lat, H_lat, W_lat)

    # 7. Start from pure noise
    latents = torch.randn_like(lat)
    fixed_frame = 1

    pipe.scheduler.set_timesteps(num_inference_steps, shift=5.0)

    for progress_id, ts in enumerate(pipe.scheduler.timesteps):
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
    """Save list of PIL Image frames to MP4."""
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()


def mux_video_with_audio(frames, audio_path, output_path, fps=25):
    """Save video and mux with audio via ffmpeg."""
    temp_video = output_path.replace(".mp4", "_temp_nosound.mp4")
    save_video_frames(frames, temp_video, fps=fps)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_video, "-i", audio_path,
             "-c:v", "copy", "-c:a", "aac", "-shortest", output_path],
            capture_output=True, check=True,
        )
    except Exception:
        # If mux fails, just use the video without audio
        shutil.copy2(temp_video, output_path)
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)


def load_syncnet(args):
    """Lazy-load SyncNet for validation metrics. Returns (syncnet, detector) or (None, None)."""
    if not getattr(args, "compute_sync_metrics", False):
        return None, None
    try:
        # Add DiffSynth-Studio to path for syncnet imports
        diffsynth_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "DiffSynth-Studio",
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
    """Run SyncNet evaluation on a video. Returns (sync_d, sync_c) or (None, None)."""
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


def run_validation_loop(model, val_samples, dataset, args, global_step, device,
                        syncnet=None, syncnet_detector=None, use_wandb=False):
    """Run validation on preloaded samples, log to wandb."""
    unwrapped = model
    val_temp_dir = tempfile.mkdtemp(prefix="val_omniavatar_")
    val_videos = []
    sync_d_list, sync_c_list = [], []

    for idx, sample in enumerate(val_samples):
        try:
            print(f"[Validation] Generating sample {idx + 1}/{len(val_samples)}...")
            generated_frames = run_validation(
                unwrapped, sample, device,
                num_inference_steps=args.val_num_inference_steps,
                guidance_scale=args.val_cfg_scale,
                negative_prompt=getattr(args, "negative_prompt", ""),
            )

            video_path = os.path.join(val_temp_dir, f"val_{global_step}_{idx}.mp4")
            audio_path = sample.get("video_path", "")

            if audio_path and os.path.exists(audio_path):
                mux_video_with_audio(generated_frames, audio_path, video_path, fps=25)
            else:
                save_video_frames(generated_frames, video_path, fps=25)

            # SyncNet metrics
            sync_d, sync_c = None, None
            if syncnet is not None and syncnet_detector is not None:
                sync_d, sync_c = run_syncnet_eval(syncnet, syncnet_detector, video_path, val_temp_dir)
                if sync_d is not None:
                    sync_d_list.append(sync_d)
                    sync_c_list.append(sync_c)
                    print(f"[SyncNet] Sample {idx}: Sync-D={sync_d:.2f}, Sync-C={sync_c:.2f}")

            # Collect wandb videos
            if use_wandb:
                import wandb
                caption = f"step_{global_step}_{idx}"
                if sync_d is not None:
                    caption += f"_D{sync_d:.1f}_C{sync_c:.1f}"
                val_videos.append(wandb.Video(video_path, format="mp4", caption=caption))

        except Exception as e:
            print(f"[Validation] Error on sample {idx}: {e}")
            import traceback
            traceback.print_exc()

    # Log to wandb
    if use_wandb and val_videos:
        try:
            import wandb
            wandb.log({"val/videos": val_videos}, step=global_step)
            if sync_d_list:
                avg_d = sum(sync_d_list) / len(sync_d_list)
                avg_c = sum(sync_c_list) / len(sync_c_list)
                wandb.log({"val/sync_d": avg_d, "val/sync_c": avg_c}, step=global_step)
                print(f"[Validation] Avg Sync-D: {avg_d:.2f}, Avg Sync-C: {avg_c:.2f}")
        except Exception as e:
            print(f"[Validation] wandb log error: {e}")

    # Cleanup
    shutil.rmtree(val_temp_dir, ignore_errors=True)


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(accelerator, model, output_path, step_or_name, train_args):
    """Save trainable weights only (LoRA + audio modules + patch_embedding)."""
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

        # Save config.json
        config = {
            "model_config": {"in_dim": 33, "audio_hidden_size": 32},
            "train_architecture": "lora",
            "lora_rank": train_args.lora_rank,
            "lora_alpha": train_args.lora_alpha,
            "lora_target_modules": train_args.lora_target_modules,
            "init_lora_weights": "kaiming",
            "use_audio": True,
            "i2v": True,
            "random_prefix_frames": True,
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)


# ============================================================================
# Training Loop
# ============================================================================

def launch_training(dataset, model, args):
    """Main training loop with Accelerate, wandb, gradient accumulation, validation."""
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

    # Move auxiliary loss models to device
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

    # --- Preload validation samples ---
    val_samples = []
    if args.validation_steps > 0 and accelerator.is_main_process:
        num_val = min(args.num_val_samples, len(dataset))
        print(f"[Validation] Loading {num_val} validation samples...")
        for idx in range(num_val):
            try:
                sample = dataset[idx]
                if sample is not None:
                    val_samples.append(sample)
            except Exception as e:
                print(f"[Validation] Failed to load sample {idx}: {e}")
        print(f"[Validation] Loaded {len(val_samples)} samples")

    # --- SyncNet ---
    syncnet, syncnet_detector = None, None
    if accelerator.is_main_process:
        syncnet, syncnet_detector = load_syncnet(args)

    # --- wandb ---
    use_wandb = getattr(args, "use_wandb", False) and accelerator.is_main_process
    if use_wandb:
        try:
            import wandb
            api_key = getattr(args, "wandb_api_key", None)
            if api_key:
                wandb.login(key=api_key, relogin=True)
            wandb.init(
                project=getattr(args, "wandb_project", "OmniAvatar-Training"),
                entity=getattr(args, "wandb_entity", None),
                name=getattr(args, "wandb_run_name", None),
                tags=args.wandb_tags.split(",") if getattr(args, "wandb_tags", None) else None,
                config={
                    "learning_rate": args.learning_rate,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                    "num_frames": args.num_frames, "height": args.height, "width": args.width,
                    "model": "OmniAvatar-14B",
                },
            )
            print(f"[W&B] Initialized: {wandb.run.id}")
        except Exception as e:
            print(f"[W&B] Init failed: {e}")
            use_wandb = False

    # --- Loss tracking ---
    ema_loss = None
    ema_beta = 0.99
    window_loss_sum = window_loss_count = 0.0
    cum_loss_sum = cum_loss_count = 0.0
    ga_loss_sum = ga_loss_count = 0.0
    global_step = 0

    # --- Validation at step 0 ---
    if getattr(args, "validate_at_start", False) and accelerator.is_main_process and val_samples:
        print("[Validation] Running validation at step 0 (baseline)...")
        unwrapped = accelerator.unwrap_model(model)
        device = next(unwrapped.dit.parameters()).device
        run_validation_loop(
            unwrapped, val_samples, dataset, args, global_step, device,
            syncnet, syncnet_detector, use_wandb,
        )

    # --- Training loop ---
    consecutive_nones = 0
    for epoch_id in range(args.num_epochs):
        epoch_pbar = tqdm(
            dataloader, desc=f"Epoch {epoch_id + 1}/{args.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for data in epoch_pbar:
            if data is None:
                consecutive_nones += 1
                if consecutive_nones >= 50:
                    raise RuntimeError(
                        f"50 consecutive None samples — check dataset path and video files"
                    )
                continue
            consecutive_nones = 0

            with accelerator.accumulate(model):
                optimizer.zero_grad()
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

                    # wandb logging
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
                            # Auxiliary loss components
                            uw = accelerator.unwrap_model(model)
                            if uw._last_aux_losses:
                                for k, v in uw._last_aux_losses.items():
                                    log_dict[f"aux/{k}"] = v.item() if hasattr(v, 'item') else v
                            wandb.log(log_dict, step=global_step)
                        except Exception as e:
                            print(f"[W&B] log error: {e}")
                        window_loss_sum = window_loss_count = 0.0

                    # Checkpoint
                    if args.save_steps and global_step % args.save_steps == 0:
                        save_checkpoint(accelerator, model, args.output_path, global_step, args)

                    lr_scheduler.step()

                    # Validation
                    if (
                        args.validation_steps > 0
                        and global_step % args.validation_steps == 0
                        and accelerator.is_main_process
                        and val_samples
                    ):
                        print(f"\n[Validation] Running at step {global_step}...")
                        unwrapped = accelerator.unwrap_model(model)
                        device = next(unwrapped.dit.parameters()).device
                        run_validation_loop(
                            unwrapped, val_samples, dataset, args, global_step, device,
                            syncnet, syncnet_detector, use_wandb,
                        )

    # Final checkpoint
    save_checkpoint(accelerator, model, args.output_path, f"final-{global_step}", args)

    if use_wandb:
        try:
            import wandb
            wandb.summary["final/loss_ema"] = ema_loss
            wandb.summary["final/total_steps"] = global_step
            wandb.finish()
        except Exception:
            pass


# ============================================================================
# Argument Parser
# ============================================================================

def train_parser():
    parser = argparse.ArgumentParser(description="OmniAvatar 14B training script.")

    # Model paths
    parser.add_argument("--dit_paths", type=str, required=True)
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--wav2vec_path", type=str, required=True)
    parser.add_argument("--omniavatar_ckpt", type=str, default=None,
                        help="Path to OmniAvatar checkpoint for finetuning. Omit for from-scratch.")

    # Dataset
    parser.add_argument("--dataset_base_path", type=str, required=True)
    parser.add_argument("--dataset_metadata_path", type=str, required=True)
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
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping. 0 to disable.")

    # Condition dropout for CFG training
    parser.add_argument("--text_drop_prob", type=float, default=0.1,
                        help="Probability of dropping text condition (replacing with empty encoding)")
    parser.add_argument("--audio_drop_prob", type=float, default=0.1,
                        help="Probability of dropping audio condition (replacing with zeros)")

    # VRAM management
    parser.add_argument("--offload_frozen", action="store_true", default=False,
                        help="Offload frozen encoders (T5, VAE, Wav2Vec) to CPU between uses. "
                             "Saves ~12 GB VRAM but adds latency per step.")
    parser.add_argument("--verbose_vram", action="store_true", default=False,
                        help="Print VRAM usage at each stage of the forward pass.")

    # Checkpoint
    parser.add_argument("--output_path", type=str, default="./checkpoints/omniavatar-14b")
    parser.add_argument("--save_steps", type=int, default=None)

    # Wandb
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="OmniAvatar-Training")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--wandb_log_every", type=int, default=10)

    # Validation
    parser.add_argument("--validation_steps", type=int, default=0)
    parser.add_argument("--num_val_samples", type=int, default=4)
    parser.add_argument("--val_num_inference_steps", type=int, default=25)
    parser.add_argument("--val_cfg_scale", type=float, default=4.5)
    parser.add_argument("--validate_at_start", action="store_true", default=False)

    # SyncNet (validation metrics)
    parser.add_argument("--compute_sync_metrics", action="store_true", default=False)
    parser.add_argument("--syncnet_model_path", type=str,
                        default="/home/work/.local/LatentSync/checkpoints/auxiliary/syncnet_v2.model")
    parser.add_argument("--s3fd_model_path", type=str,
                        default="/home/work/.local/LatentSync/checkpoints/auxiliary/sfd_face.pth")

    # Auxiliary losses (SyncNet, LPIPS, TREPA on VAE-decoded x_0)
    parser.add_argument("--use_sync_loss", action="store_true", default=False,
                        help="Enable SyncNet audio-visual sync training loss")
    parser.add_argument("--use_lpips_loss", action="store_true", default=False,
                        help="Enable LPIPS perceptual training loss (lower half)")
    parser.add_argument("--use_trepa_loss", action="store_true", default=False,
                        help="Enable TREPA temporal consistency training loss")
    parser.add_argument("--aux_recon_weight", type=float, default=1.0)
    parser.add_argument("--aux_sync_weight", type=float, default=0.1)
    parser.add_argument("--aux_lpips_weight", type=float, default=0.1)
    parser.add_argument("--aux_trepa_weight", type=float, default=10.0)
    parser.add_argument("--aux_num_frames", type=int, default=21,
                        help="Latent frames to decode for aux losses. 21 → 81 RGB frames. 0 = all.")
    parser.add_argument("--sync_chunk_size", type=int, default=16)
    parser.add_argument("--sync_chunk_stride", type=int, default=8)
    parser.add_argument("--sync_num_supervised_frames", type=int, default=80)
    parser.add_argument("--use_vae_gradient_checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing in VAE decoder (for aux losses)")
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
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config_yaml(args):
    """Load YAML config and use values as defaults for unset args."""
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

    # Dataset
    dataset = OmniAvatarTrainingDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        num_frames=args.num_frames,
        height=args.height, width=args.width,
        sample_rate=args.sample_rate, fps=args.fps,
        repeat=args.dataset_repeat,
    )
    print(f"[Dataset] {len(dataset)} samples")

    # Model
    model = OmniAvatarTrainingModule(
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
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total: {total:,} | Trainable: {trainable:,} ({trainable/total*100:.2f}%)")

    # Setup auxiliary losses (before accelerator.prepare)
    if args.use_sync_loss or args.use_lpips_loss or args.use_trepa_loss:
        model.setup_aux_losses(args)

    # Train
    launch_training(dataset, model, args)
