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
from accelerate.utils import DistributedDataParallelKwargs
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
    # Forward pass — flow matching training loss
    # ------------------------------------------------------------------

    def forward(self, data):
        """Training forward pass. Must match inference DiT forward exactly."""
        device = next(self.dit.parameters()).device

        # 1. Encode text
        context = self.encode_text(data["prompt"], device)

        # 2. Encode audio
        num_video_frames = len(data["video"])
        audio_emb = self.encode_audio(data["audio"], num_video_frames, device)
        audio_emb = audio_emb.to(dtype=self.dtype, device=device)

        # 3. Encode GT video
        input_latents = self.encode_video(data["video"], device)

        # 4. Prepare reference frame
        ref_latent = input_latents[:, :, :1]
        T_lat = input_latents.shape[2]
        y = self.prepare_reference_input(ref_latent, T_lat)

        # 5. Sample noise + timestep
        noise = torch.randn_like(input_latents)
        num_timesteps = len(self.scheduler.timesteps)
        timestep_id = torch.randint(0, num_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=device)

        noisy_latents = self.scheduler.add_noise(input_latents, noise, timestep)
        training_target = self.scheduler.training_target(input_latents, noise, timestep)

        # 6. DiT forward (THE critical call)
        noise_pred = self.dit(
            x=noisy_latents, timestep=timestep, context=context, y=y,
            audio_emb=audio_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )

        # 7. Flow matching loss
        loss = F.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
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

    # 1. Encode text
    context = model.encode_text(val_sample["prompt"], device)

    # 2. Encode audio
    audio_emb = model.encode_audio(val_sample["audio"], num_video_frames, device)
    audio_emb = audio_emb.to(dtype=model.dtype, device=device)

    # 3. Encode reference image as SINGLE frame (matching inference.py:270)
    ref_image = val_sample["ref_image"]
    ref_np = np.array(ref_image, dtype=np.float32) * (2.0 / 255.0) - 1.0
    ref_t = torch.from_numpy(ref_np).permute(2, 0, 1)  # (C, H, W)
    ref_video = ref_t.unsqueeze(1).unsqueeze(0)  # (1, C, 1, H, W)
    model.vae.to(device)
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
    # lat = [ref_latent, zeros for remaining frames]
    lat = torch.cat([
        ref_latent,
        torch.zeros(1, 16, T_lat - 1, ref_latent.shape[3], ref_latent.shape[4],
                     dtype=model.dtype, device=device),
    ], dim=2)  # (1, 16, T_lat, H_lat, W_lat)

    # 7. Start from pure noise
    latents = torch.randn_like(lat)
    fixed_frame = 1  # first_fixed_frame = 1 in inference (first chunk)

    # Negative text for CFG
    neg_context = model.encode_text(negative_prompt, device) if guidance_scale != 1.0 else None

    pipe.scheduler.set_timesteps(num_inference_steps, shift=5.0)

    for progress_id, ts in enumerate(pipe.scheduler.timesteps):
        # Fixed-frame overwriting (matching inference wan_video.py:256-257)
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

    # Final fixed-frame overwrite
    if fixed_frame > 0:
        latents[:, :, :fixed_frame] = lat[:, :, :fixed_frame]

    # Decode
    frames = pipe.decode_video(latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
    pipe.device = old_device
    if frames.dim() == 5:
        frames = frames[0]
    video_frames = pipe.tensor2video(frames)

    # Restore scheduler
    pipe.scheduler.set_timesteps(1000, training=True)
    model.train()
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
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
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
                    ga_loss_sum += loss.item()
                    ga_loss_count += 1

                if accelerator.sync_gradients:
                    step_loss = ga_loss_sum / max(1, ga_loss_count)
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
                            wandb.log({
                                "loss/step": step_loss, "loss/ema": ema_loss,
                                "loss/window_mean": window_loss_sum / max(1, window_loss_count),
                                "loss/cum_mean": cum_loss_sum / max(1, cum_loss_count),
                                "train/epoch": epoch_id,
                                "train/lr": lr_scheduler.get_last_lr()[0],
                            }, step=global_step)
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

    # SyncNet
    parser.add_argument("--compute_sync_metrics", action="store_true", default=False)
    parser.add_argument("--syncnet_model_path", type=str,
                        default="/home/work/.local/LatentSync/checkpoints/auxiliary/syncnet_v2.model")
    parser.add_argument("--s3fd_model_path", type=str,
                        default="/home/work/.local/LatentSync/checkpoints/auxiliary/sfd_face.pth")

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

    # Train
    launch_training(dataset, model, args)
