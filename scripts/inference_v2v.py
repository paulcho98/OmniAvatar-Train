"""OmniAvatar V2V inference script for video-to-video lip sync.

Supports two modes:
1. Standard mode (default): Pre-aligned 512x512 source video.
2. LatentSync mode (-hp latentsync_inference=true): Arbitrary-resolution video
   with InsightFace face detection, affine alignment, generation on aligned face,
   and inverse affine compositing back onto the original frame.

Input format (@@-delimited): prompt@@video_path@@audio_path

Usage:
    # Standard mode (pre-aligned 512x512 data)
    torchrun --standalone --nproc_per_node=1 scripts/inference_v2v.py \\
        --config configs/inference_v2v.yaml --input_file examples/v2v_samples.txt

    # LatentSync mode (arbitrary-resolution video)
    torchrun --standalone --nproc_per_node=1 scripts/inference_v2v.py \\
        --config configs/inference_v2v.yaml --input_file examples/v2v_samples.txt \\
        -hp latentsync_inference=true

    # Use a specific checkpoint step
    torchrun --standalone --nproc_per_node=1 scripts/inference_v2v.py \\
        --config configs/inference_v2v.yaml --input_file examples/v2v_samples.txt \\
        -hp ckpt_name=step-500.pt

Based on inference_modified.py with V2V-specific changes:
- 49-channel input (16 noise + 16 ref + 1 spatial_mask + 16 masked_video)
- Source video provides structure; model regenerates mouth region from audio
- Single-pass generation (81 frames / 21 latent frames), no chunking
"""
import subprocess
import os, sys
import logging
from glob import glob
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import random
import shutil
import tempfile
import librosa
import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial
from PIL import Image
from OmniAvatar.utils.args_config import parse_args
args = parse_args()

from OmniAvatar.utils.io_utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.wan_video import WanVideoPipeline
from OmniAvatar.utils.io_utils import save_video_as_grid_and_mp4
import torch.distributed as dist
import torchvision.transforms as TT
from transformers import Wav2Vec2FeatureExtractor
import torchvision.transforms as transforms
import torch.nn.functional as F
from OmniAvatar.utils.audio_preprocess import add_silence_to_audio_ffmpeg
from OmniAvatar.distributed.fsdp import shard_model

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_from_file(p):
    with open(p, "r") as fin:
        for l in fin:
            yield l.strip()


# ---------------------------------------------------------------------------
# LatentSync preprocessing / compositing (adapted from StableAvatar)
# ---------------------------------------------------------------------------

def preprocess_with_latentsync(video_path, image_processor, face_detection_cache_dir, num_frames=81):
    """Detect faces, align to 512x512 via affine transform, with caching."""
    if not os.path.exists(video_path):
        logger.warning(f"[LatentSync] Video not found: {video_path}")
        return None

    try:
        entry_dir_name = os.path.basename(os.path.dirname(video_path))
        video_stem = entry_dir_name if entry_dir_name else os.path.splitext(os.path.basename(video_path))[0]
        face_cache_path = os.path.join(face_detection_cache_dir, f"{video_stem}_face_cache.pt")

        face_cache_loaded = False
        original_frames = None
        if os.path.isfile(face_cache_path):
            try:
                face_cache = torch.load(face_cache_path, weights_only=False)
                if face_cache.get("resolution") == image_processor.resolution:
                    boxes = face_cache["boxes"]
                    affine_matrices = face_cache["affine_matrices"]
                    aligned_faces = face_cache["aligned_faces"]
                    detection_failures = []
                    face_cache_loaded = True
                    print(f"[LatentSync] Loaded face cache: {face_cache_path}")
                else:
                    print(f"[LatentSync] Cache stale, recomputing...")
            except Exception as e:
                print(f"[LatentSync] Cache corrupt ({e}), recomputing...")
                os.remove(face_cache_path)

        if not face_cache_loaded:
            cap = cv2.VideoCapture(video_path)
            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()

            if len(frames) < 5:
                print(f"[LatentSync] Too few frames ({len(frames)}) in {video_path}")
                return None

            while len(frames) < num_frames:
                frames.append(frames[-1].copy())

            original_frames = np.stack(frames, axis=0)
            boxes = []
            affine_matrices = []
            aligned_faces = []
            detection_failures = []

            # Reset temporal smoothing bias for new video
            image_processor.restorer.p_bias = None

            for i, frame in enumerate(frames):
                try:
                    face, box, affine_matrix = image_processor.affine_transform(frame)
                    boxes.append(box)
                    affine_matrices.append(affine_matrix)
                    aligned_faces.append(face)
                except RuntimeError as e:
                    print(f"[LatentSync] Face detection failed for frame {i}: {e}")
                    boxes.append(None)
                    affine_matrices.append(None)
                    detection_failures.append(i)

            if detection_failures:
                print(f"[LatentSync] Face detection failed for {len(detection_failures)} frames, skipping")
                return None

            os.makedirs(face_detection_cache_dir, exist_ok=True)
            torch.save({
                "aligned_faces": aligned_faces,
                "boxes": boxes,
                "affine_matrices": affine_matrices,
                "resolution": image_processor.resolution,
                "num_frames": len(original_frames),
            }, face_cache_path)
            print(f"[LatentSync] Saved face cache: {face_cache_path}")

        return {
            "video_path": video_path,
            "original_frames": original_frames,
            "num_frames": num_frames,
            "aligned_faces": aligned_faces,
            "boxes": boxes,
            "affine_matrices": affine_matrices,
            "detection_failures": detection_failures if not face_cache_loaded else [],
        }

    except Exception as e:
        print(f"[LatentSync] Preprocessing failed for {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def composite_with_latentsync(generated_frames_np, latentsync_metadata, image_processor,
                               use_mouth_only_compositing=False):
    """Composite generated aligned faces back onto original video frames."""
    import torchvision.transforms.functional as TF_v

    original_frames = latentsync_metadata["original_frames"]
    if original_frames is None:
        video_path = latentsync_metadata["video_path"]
        num_frames = latentsync_metadata.get("num_frames", 81)
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
        original_frames = np.stack(frames, axis=0)

    boxes = latentsync_metadata["boxes"]
    affine_matrices = latentsync_metadata["affine_matrices"]
    detection_failures = latentsync_metadata.get("detection_failures", [])
    aligned_faces = latentsync_metadata.get("aligned_faces", None)

    composite_frames = []

    for i in range(generated_frames_np.shape[0]):
        if i in detection_failures or boxes[i] is None:
            composite_frames.append(original_frames[i])
            continue

        face_tensor = torch.from_numpy(generated_frames_np[i]).permute(2, 0, 1)  # HWC -> CHW

        if use_mouth_only_compositing and aligned_faces is not None:
            original_aligned_face = aligned_faces[i]
            mouth_mask = image_processor.mask_image
            face_tensor_float = face_tensor.float()
            composited_face = (
                face_tensor_float * (1 - mouth_mask) +
                original_aligned_face.float() * mouth_mask
            )
            face_tensor = composited_face.byte()

        x1, y1, x2, y2 = boxes[i]
        height = int(y2 - y1)
        width = int(x2 - x1)

        face_resized = TF_v.resize(
            face_tensor, size=[height, width],
            interpolation=TF_v.InterpolationMode.BICUBIC, antialias=True,
        )

        face_resized = face_resized.float() / 255.0
        face_resized = face_resized * 2.0 - 1.0

        try:
            restored_frame = image_processor.restorer.restore_img(
                original_frames[i], face_resized, affine_matrices[i]
            )
            composite_frames.append(restored_frame)
        except Exception as e:
            print(f"[LatentSync] Restoration failed for frame {i}: {e}")
            composite_frames.append(original_frames[i])

    return np.stack(composite_frames)


# ---------------------------------------------------------------------------
# Video loading utilities
# ---------------------------------------------------------------------------

def load_video_frames(video_path, num_frames=81, target_size=(512, 512)):
    """Load video frames, resize to target_size, pad to num_frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Could not read any frames from {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return np.stack(frames, axis=0)  # [N, H, W, 3] uint8


def apply_spatial_mask(frames_np, mask_np):
    """Apply LatentSync mask: frame 0 untouched, frames 1+ have mouth zeroed.

    Args:
        frames_np: [N, H, W, 3] uint8
        mask_np: [H, W] float32, 1=keep, 0=mask (LatentSync convention)
    Returns:
        masked_frames_np: [N, H, W, 3] uint8
    """
    masked = frames_np.copy()
    for i in range(1, len(masked)):
        masked[i] = (masked[i].astype(np.float32) * mask_np[:, :, None]).astype(np.uint8)
    return masked


def frames_to_video_tensor(frames_np):
    """Convert [N, H, W, 3] uint8 numpy to [1, 3, N, H, W] float [-1, 1] tensor."""
    t = torch.from_numpy(frames_np).float() / 255.0  # [N, H, W, 3] in [0, 1]
    t = t.permute(0, 3, 1, 2)  # [N, 3, H, W]
    t = t * 2.0 - 1.0  # [-1, 1]
    t = t.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, N, H, W]
    return t


def load_latentsync_mask(mask_path, latent_h, latent_w):
    """Load LatentSync mask and resize to latent resolution.

    Returns [H_lat, W_lat] float tensor, 1=keep, 0=mask (LatentSync convention).
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_arr = np.array(mask_img).astype(np.float32) / 255.0  # 1=keep, 0=mask
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    mask_resized = F.interpolate(mask_t, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
    return (mask_resized > 0.5).float().squeeze(0).squeeze(0)  # [H_lat, W_lat]


def prepare_v2v_y(ref_latent, masked_video_latents, latent_mask):
    """Build 33-channel y tensor for V2V (from train_v2v.py:prepare_v2v_input).

    Args:
        ref_latent: [1, 16, 1, H, W]
        masked_video_latents: [1, 16, T, H, W]
        latent_mask: [H, W] (1=keep, 0=mask, LatentSync convention)
    Returns:
        y: [1, 33, T, H, W]
    """
    T_lat = masked_video_latents.shape[2]
    H_lat, W_lat = masked_video_latents.shape[3], masked_video_latents.shape[4]
    device = masked_video_latents.device
    dtype = masked_video_latents.dtype

    ref_repeated = ref_latent.repeat(1, 1, T_lat, 1, 1)

    # OmniAvatar convention: 0=keep, 1=generate. Invert from LatentSync.
    inverted = 1.0 - latent_mask
    mask_ch = torch.zeros(1, 1, T_lat, H_lat, W_lat, device=device, dtype=dtype)
    mask_ch[:, :, 0] = 0  # Frame 0: keep all (reference)
    mask_ch[:, :, 1:] = inverted[None, None, None]

    return torch.cat([ref_repeated, mask_ch, masked_video_latents], dim=1)


# ---------------------------------------------------------------------------
# Mux video with audio via ffmpeg
# ---------------------------------------------------------------------------

def mux_video_with_audio(video_path_silent, audio_path, output_path, duration_s=None):
    """Mux a silent video with an audio file using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path_silent, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-shortest",
    ]
    if duration_s is not None:
        cmd.extend(["-t", f"{duration_s:.4f}"])
    cmd.append(output_path)
    subprocess.run(cmd, capture_output=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class WanV2VInferencePipeline(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(f"cuda:{args.rank}")
        if args.dtype == 'bf16':
            self.dtype = torch.bfloat16
        elif args.dtype == 'fp16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.pipe = self.load_model()

        # Audio encoder (float32 — bf16 breaks CNN feature extractor)
        if args.use_audio:
            from OmniAvatar.models.wav2vec import Wav2VecModel
            self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)
            self.audio_encoder = Wav2VecModel.from_pretrained(
                args.wav2vec_path, local_files_only=True
            ).to(device=self.device)
            self.audio_encoder.feature_extractor._freeze_parameters()

        # LatentSync mask
        self.latentsync_mask_path = getattr(args, "latentsync_mask_path", None)

        # LatentSync face processor (only for --latentsync_inference)
        self.image_processor = None
        if getattr(args, "latentsync_inference", False):
            from OmniAvatar.utils.latentsync import ImageProcessor
            insightface_root = getattr(args, "insightface_model_dir", "checkpoints/auxiliary")
            mask_image_path = self.latentsync_mask_path
            self.image_processor = ImageProcessor(
                resolution=512,
                device=str(self.device),
                mask_image=None,  # will load default mask.png
                insightface_root=insightface_root,
            )

    def load_model(self):
        dist.init_process_group(backend="nccl", init_method="env://")
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=args.sp_size,
            ring_degree=1,
            ulysses_degree=args.sp_size,
        )
        torch.cuda.set_device(dist.get_rank())

        # Resolve checkpoint path
        ckpt_name = getattr(args, "ckpt_name", "pytorch_model.pt")
        ckpt_path = os.path.join(args.exp_path, ckpt_name)
        assert os.path.exists(ckpt_path), f"{ckpt_name} not found in {args.exp_path}"
        if args.train_architecture == 'lora':
            pretrained_lora_path = ckpt_path
        else:
            resume_path = ckpt_path

        # Load base models
        model_manager = ModelManager(device="cpu", infer=True)
        model_manager.load_models(
            [args.dit_path.split(","), args.text_encoder_path, args.vae_path],
            torch_dtype=self.dtype, device='cpu',
        )

        pipe = WanVideoPipeline.from_model_manager(
            model_manager, torch_dtype=self.dtype,
            device=f"cuda:{dist.get_rank()}",
            use_usp=True if args.sp_size > 1 else False,
            infer=True,
        )

        # Load trained checkpoint
        if args.train_architecture == "lora":
            print(f'Use LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}')
            self.add_lora_to_model(
                pipe.denoising_model(),
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_target_modules=args.lora_target_modules,
                init_lora_weights=args.init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            missing, unexpected = pipe.denoising_model().load_state_dict(
                load_state_dict(resume_path), strict=True
            )
            print(f"Loaded {resume_path}: {len(missing)} missing, {len(unexpected)} unexpected keys")

        pipe.requires_grad_(False)
        pipe.eval()
        pipe.enable_vram_management(
            num_persistent_param_in_dit=args.num_persistent_param_in_dit
        )
        if args.use_fsdp:
            shard_fn = partial(shard_model, device_id=self.device)
            pipe.dit = shard_fn(pipe.dit)
        return pipe

    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4,
                          lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                          init_lora_weights="kaiming", pretrained_lora_path=None,
                          state_dict_converter=None):
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)

        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated = len(all_keys) - len(missing_keys)
            print(f"{num_updated} params loaded from {pretrained_lora_path}. "
                  f"{len(unexpected_keys)} unexpected.")

    @torch.no_grad()
    def encode_audio(self, audio_path, num_frames):
        """Encode audio → [1, num_frames, 10752] float tensor."""
        wav2vec_sr = 16000  # Wav2Vec2 native sample rate
        audio, sr = librosa.load(audio_path, sr=wav2vec_sr)
        input_values = np.squeeze(
            self.wav_feature_extractor(audio, sampling_rate=wav2vec_sr).input_values
        )
        input_values = torch.from_numpy(input_values).float().to(device=self.device)
        input_values = input_values.unsqueeze(0)

        # Pad audio to match video frame count
        samples_per_frame = wav2vec_sr // self.args.fps  # 640 at 16kHz/25fps
        target_samples = num_frames * samples_per_frame
        if input_values.shape[1] < target_samples:
            input_values = F.pad(input_values, (0, target_samples - input_values.shape[1]))
        elif input_values.shape[1] > target_samples:
            input_values = input_values[:, :target_samples]

        hidden_states = self.audio_encoder(
            input_values, seq_len=num_frames, output_hidden_states=True
        )
        audio_emb = hidden_states.last_hidden_state
        for hs in hidden_states.hidden_states:
            audio_emb = torch.cat((audio_emb, hs), -1)
        return audio_emb  # [1, num_frames, 10752]

    @torch.no_grad()
    def forward(self, prompt, video_path, audio_path,
                num_frames=81, num_steps=None, negative_prompt=None,
                guidance_scale=None, latentsync_metadata=None):
        num_steps = num_steps if num_steps is not None else self.args.num_steps
        negative_prompt = negative_prompt if negative_prompt is not None else self.args.negative_prompt
        guidance_scale = guidance_scale if guidance_scale is not None else self.args.guidance_scale

        H, W = 512, 512
        T_lat = (num_frames + 3) // 4  # latent temporal frames (81 → 21)
        latent_h, latent_w = H // 8, W // 8  # 64, 64

        # --- 1. Load source video frames ---
        if latentsync_metadata is not None:
            # LatentSync mode: use pre-aligned 512x512 faces
            aligned_faces = latentsync_metadata["aligned_faces"]
            frames_np = np.stack([
                cv2.resize(
                    f.permute(1, 2, 0).numpy().astype(np.uint8) if isinstance(f, torch.Tensor) else f,
                    (W, H), interpolation=cv2.INTER_LANCZOS4,
                )
                for f in aligned_faces
            ], axis=0)  # [N, H, W, 3] uint8
        else:
            # Standard mode: load pre-aligned 512x512 video
            frames_np = load_video_frames(video_path, num_frames, (W, H))

        # --- 2. Load and apply spatial mask ---
        mask_path = self.latentsync_mask_path
        if mask_path is None:
            mask_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..",
                "OmniAvatar", "utils", "latentsync", "mask.png"
            )
        mask_img = Image.open(mask_path).convert("L")
        mask_pixel = np.array(mask_img.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
        mask_pixel_binary = (mask_pixel > 0.5).astype(np.float32)

        masked_frames_np = apply_spatial_mask(frames_np, mask_pixel_binary)

        # --- 3. Convert to tensors and VAE-encode ---
        video_tensor = frames_to_video_tensor(frames_np)  # [1, 3, N, H, W] float [-1, 1]
        masked_video_tensor = frames_to_video_tensor(masked_frames_np)

        self.pipe.load_models_to_device(['vae'])

        source_latents = self.pipe.encode_video(
            video_tensor.to(dtype=self.dtype, device=self.device)
        ).to(self.device)  # [1, 16, T_lat, latent_h, latent_w]

        masked_video_latents = self.pipe.encode_video(
            masked_video_tensor.to(dtype=self.dtype, device=self.device)
        ).to(self.device)  # [1, 16, T_lat, latent_h, latent_w]

        # --- 4. Build V2V y tensor (33ch) ---
        ref_latent = source_latents[:, :, :1]  # [1, 16, 1, H, W]
        latent_mask = load_latentsync_mask(mask_path, latent_h, latent_w).to(self.device)
        y = prepare_v2v_y(ref_latent, masked_video_latents, latent_mask)

        # --- 5. Encode text (load to GPU, then offload after) ---
        self.pipe.load_models_to_device(['text_encoder'])
        prompt_emb_posi = self.pipe.encode_prompt(prompt, positive=True)
        prompt_emb_nega = self.pipe.encode_prompt(
            negative_prompt or "", positive=False
        ) if guidance_scale != 1.0 else None
        self.pipe.load_models_to_device([])

        # --- 6. Encode audio ---
        audio_emb = self.encode_audio(audio_path, num_frames)
        audio_emb = audio_emb.to(dtype=self.dtype, device=self.device)

        # --- 7. Denoising loop (single-pass, matches train_v2v validation) ---
        # Clean latent for fixed-frame overwriting (frame 0 = reference)
        lat = torch.cat([
            ref_latent,
            torch.zeros(1, 16, T_lat - 1, latent_h, latent_w,
                         dtype=self.dtype, device=self.device),
        ], dim=2)

        latents = torch.randn_like(lat)
        fixed_frame = 1

        self.pipe.scheduler.set_timesteps(num_steps, shift=5.0)
        self.pipe.load_models_to_device(["dit"])

        image_emb = {"y": y}
        audio_emb_dict = {"audio_emb": audio_emb}

        for ts in tqdm(self.pipe.scheduler.timesteps, desc="Denoising", disable=dist.get_rank() != 0):
            if fixed_frame > 0:
                latents[:, :, :fixed_frame] = lat[:, :, :fixed_frame]

            t = ts.unsqueeze(0).to(dtype=self.dtype, device=self.device)

            noise_pred_posi = self.pipe.dit(
                x=latents, timestep=t,
                **prompt_emb_posi, **image_emb, **audio_emb_dict,
            )

            if guidance_scale != 1.0:
                audio_emb_uc = {"audio_emb": torch.zeros_like(audio_emb)}
                noise_pred_nega = self.pipe.dit(
                    x=latents, timestep=t,
                    **prompt_emb_nega, **image_emb, **audio_emb_uc,
                )
                noise_pred = noise_pred_nega + guidance_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            latents = self.pipe.scheduler.step(noise_pred, ts, latents)

        # Final fixed-frame overwrite
        if fixed_frame > 0:
            latents[:, :, :fixed_frame] = lat[:, :, :fixed_frame]

        # --- 8. Decode ---
        self.pipe.load_models_to_device(['vae'])
        old_device = self.pipe.device
        self.pipe.device = self.device
        frames = self.pipe.decode_video(latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
        self.pipe.device = old_device
        self.pipe.load_models_to_device([])

        # Convert [-1, 1] → [0, 1] and permute to [B, T, C, H, W]
        # (matches log_video line 286 in wan_video.py)
        frames = (frames.permute(0, 2, 1, 3, 4).float() + 1) / 2

        if frames.dim() == 5:
            frames = frames[0]  # [T, C, H, W] in [0, 1]

        return frames


# ---------------------------------------------------------------------------
# SyncNet evaluation (reused from train_v2v.py)
# ---------------------------------------------------------------------------

def load_syncnet(args, output_dir):
    """Load SyncNet + S3FD for evaluation metrics."""
    if not getattr(args, "compute_sync_metrics", False):
        return None, None
    try:
        diffsynth_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "Self-Forcing_LipSync_StableAvatar",
        )
        if diffsynth_path not in sys.path:
            sys.path.insert(0, diffsynth_path)
        from diffsynth.models.syncnet import SyncNetEval, SyncNetDetector
        syncnet_model_path = getattr(args, "syncnet_model_path", None)
        s3fd_model_path = getattr(args, "s3fd_model_path", None)
        if not syncnet_model_path or not s3fd_model_path:
            print("[SyncNet] syncnet_model_path and s3fd_model_path must be set in config")
            return None, None
        syncnet = SyncNetEval(device="cuda" if torch.cuda.is_available() else "cpu")
        syncnet.loadParameters(syncnet_model_path)
        detector = SyncNetDetector(
            device="cuda" if torch.cuda.is_available() else "cpu",
            detect_results_dir=os.path.join(output_dir, "syncnet_detect"),
            s3fd_model_path=s3fd_model_path,
        )
        print("[SyncNet] Loaded successfully")
        return syncnet, detector
    except Exception as e:
        print(f"[SyncNet] Failed to load: {e}")
        return None, None


def run_syncnet_eval(syncnet, detector, video_path, temp_dir):
    """Run SyncNet evaluation on a video with audio."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(args.seed)
    data_iter = list(read_from_file(args.input_file))
    exp_name = os.path.basename(args.exp_path)
    date_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    num_frames = getattr(args, "num_frames", 81)
    latentsync_inference = getattr(args, "latentsync_inference", False)
    save_aligned = getattr(args, "save_aligned_video", False)
    use_mouth_only = getattr(args, "use_mouth_only_compositing", False)
    face_cache_dir = getattr(args, "face_detection_cache_dir", None)

    inferpipe = WanV2VInferencePipeline(args)

    if args.sp_size > 1:
        date_name = inferpipe.pipe.sp_group.broadcast_object_list([date_name])[0]

    # Output directory: use explicit output_dir if set, else auto-generate
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        output_dir = (
            f'demo_out/{exp_name}/v2v_'
            f'seed{args.seed}_step{args.num_steps}_cfg{args.guidance_scale}_'
            f'{date_name}'
        )
        if latentsync_inference:
            output_dir += "_latentsync"

    # Load SyncNet for evaluation (if requested)
    syncnet, syncnet_detector = None, None
    if dist.get_rank() == 0:
        syncnet, syncnet_detector = load_syncnet(args, output_dir)

    sync_d_all, sync_c_all = [], []

    for idx, text in tqdm(enumerate(data_iter), total=len(data_iter), desc="Samples"):
        if len(text) == 0:
            continue
        input_list = text.split("@@")
        if len(input_list) < 3:
            print(f"[Skip] Line {idx}: expected prompt@@video_path@@audio_path, got {len(input_list)} parts")
            continue
        prompt, video_path, audio_path = input_list[0], input_list[1], input_list[2]

        # LatentSync preprocessing
        latentsync_metadata = None
        if latentsync_inference:
            if face_cache_dir is None:
                face_cache_dir = os.path.join(output_dir, "face_cache")
            latentsync_metadata = preprocess_with_latentsync(
                video_path, inferpipe.image_processor, face_cache_dir, num_frames
            )
            if latentsync_metadata is None:
                print(f"[Skip] LatentSync preprocessing failed for {video_path}")
                continue

        # Generate
        frames = inferpipe(
            prompt=prompt,
            video_path=video_path,
            audio_path=audio_path,
            num_frames=num_frames,
            latentsync_metadata=latentsync_metadata,
        )

        if dist.get_rank() == 0:
            os.makedirs(output_dir, exist_ok=True)
            sample_name = os.path.splitext(os.path.basename(video_path))[0]
            video_stem = f"result_{idx:03d}_{sample_name}"

            # Convert generated frames [T, C, H, W] float [0, 1] → [T, H, W, C] uint8
            generated_np = (frames.permute(0, 2, 3, 1).clamp(0, 1) * 255).byte().cpu().numpy()

            # The video path used for SyncNet evaluation (must have audio)
            eval_video_path = None

            if latentsync_inference and latentsync_metadata is not None:
                # Save aligned video (optional)
                if save_aligned:
                    aligned_path = os.path.join(output_dir, f"{video_stem}_aligned.mp4")
                    save_frames_as_video(generated_np, aligned_path, fps=25)

                # Composite back onto original resolution
                composited_np = composite_with_latentsync(
                    generated_np, latentsync_metadata, inferpipe.image_processor,
                    use_mouth_only_compositing=use_mouth_only,
                )
                final_path = os.path.join(output_dir, f"{video_stem}_composited.mp4")
                save_frames_as_video(composited_np, final_path, fps=25)

                # Mux with audio
                final_with_audio = os.path.join(output_dir, f"{video_stem}_composited_audio.mp4")
                duration_s = num_frames / 25.0
                mux_video_with_audio(final_path, audio_path, final_with_audio, duration_s)
                eval_video_path = final_with_audio
                print(f"[Output] {final_with_audio}")
            else:
                # Standard mode: save video + mux with audio
                silent_path = os.path.join(output_dir, f"{video_stem}.mp4")
                save_frames_as_video(generated_np, silent_path, fps=25)
                with_audio_path = os.path.join(output_dir, f"{video_stem}_audio.mp4")
                duration_s = num_frames / 25.0
                mux_video_with_audio(silent_path, audio_path, with_audio_path, duration_s)
                eval_video_path = with_audio_path
                print(f"[Output] {with_audio_path}")

            # SyncNet evaluation
            if syncnet is not None and syncnet_detector is not None and eval_video_path is not None:
                sync_d, sync_c = run_syncnet_eval(
                    syncnet, syncnet_detector, eval_video_path, output_dir
                )
                if sync_d is not None:
                    sync_d_all.append(sync_d)
                    sync_c_all.append(sync_c)
                    print(f"[SyncNet] {sample_name}: Sync-D={sync_d:.2f}, Sync-C={sync_c:.2f}")

        dist.barrier()

    # Print summary
    if dist.get_rank() == 0:
        print(f"\n[Done] All outputs saved to {output_dir}")
        if sync_d_all:
            avg_d = sum(sync_d_all) / len(sync_d_all)
            avg_c = sum(sync_c_all) / len(sync_c_all)
            print(f"[SyncNet] Average Sync-D: {avg_d:.2f}, Sync-C: {avg_c:.2f} ({len(sync_d_all)} samples)")
            # Save metrics to file
            metrics_path = os.path.join(output_dir, "sync_metrics.txt")
            with open(metrics_path, "w") as f:
                f.write(f"avg_sync_d: {avg_d:.4f}\n")
                f.write(f"avg_sync_c: {avg_c:.4f}\n")
                f.write(f"num_samples: {len(sync_d_all)}\n")
                for i, (d, c) in enumerate(zip(sync_d_all, sync_c_all)):
                    f.write(f"sample_{i}: sync_d={d:.4f} sync_c={c:.4f}\n")
            print(f"[SyncNet] Metrics saved to {metrics_path}")


def save_frames_as_video(frames_np, output_path, fps=25):
    """Save [N, H, W, 3] uint8 numpy array as mp4 video."""
    import imageio
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()


class NoPrint:
    def write(self, x):
        pass
    def flush(self):
        pass


if __name__ == '__main__':
    if not args.debug:
        if args.local_rank != 0:
            sys.stdout = NoPrint()
    main()
