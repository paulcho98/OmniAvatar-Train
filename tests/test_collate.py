"""Tests for batched collate_fn."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest


def make_sample(seed=0):
    """Create a mock dataset sample matching OmniAvatarV2VDataset output."""
    torch.manual_seed(seed)
    return {
        "precomputed_input_latents": torch.randn(16, 21, 64, 64),
        "precomputed_masked_latents": torch.randn(16, 21, 64, 64),
        "precomputed_audio_emb": torch.randn(81, 10752),
        "precomputed_text_emb": torch.randn(1, 512, 4096),
        "precomputed_ref_sequence": torch.randn(16, 21, 64, 64),
        "prompt": "a person is talking",
        "video_dir": f"/data/sample_{seed}",
        "audio_path": f"/data/sample_{seed}/audio.wav",
    }


from scripts.train_v2v import batched_collate_fn


class TestBatchedCollate:
    def test_single_sample(self):
        """Batch of 1 should produce dict with batch dim."""
        s = make_sample(0)
        result = batched_collate_fn([s])
        assert result["precomputed_input_latents"].shape == (1, 16, 21, 64, 64)
        assert result["precomputed_audio_emb"].shape == (1, 81, 10752)
        assert result["precomputed_text_emb"].shape == (1, 512, 4096)
        assert result["prompt"] == ["a person is talking"]

    def test_multi_sample(self):
        """Batch of 3 should stack tensors along dim 0."""
        samples = [make_sample(i) for i in range(3)]
        result = batched_collate_fn(samples)
        assert result["precomputed_input_latents"].shape == (3, 16, 21, 64, 64)
        assert result["precomputed_masked_latents"].shape == (3, 16, 21, 64, 64)
        assert result["precomputed_audio_emb"].shape == (3, 81, 10752)
        assert result["precomputed_text_emb"].shape == (3, 512, 4096)
        assert result["precomputed_ref_sequence"].shape == (3, 16, 21, 64, 64)
        assert len(result["prompt"]) == 3
        assert len(result["video_dir"]) == 3

    def test_filters_none(self):
        """None samples should be filtered out."""
        samples = [make_sample(0), None, make_sample(2)]
        result = batched_collate_fn(samples)
        assert result["precomputed_input_latents"].shape == (2, 16, 21, 64, 64)

    def test_all_none_returns_none(self):
        result = batched_collate_fn([None, None])
        assert result is None

    def test_preserves_values(self):
        """Stacked tensors must preserve original values."""
        s0, s1 = make_sample(0), make_sample(1)
        result = batched_collate_fn([s0, s1])
        assert torch.equal(result["precomputed_input_latents"][0], s0["precomputed_input_latents"])
        assert torch.equal(result["precomputed_input_latents"][1], s1["precomputed_input_latents"])
