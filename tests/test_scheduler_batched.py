"""Tests for FlowMatchScheduler batched timestep support."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest
from OmniAvatar.schedulers.flow_match import FlowMatchScheduler


@pytest.fixture
def scheduler():
    s = FlowMatchScheduler(num_inference_steps=50, num_train_timesteps=1000, shift=3.0)
    s.set_timesteps(50, training=True)
    return s


class TestAddNoise:
    def test_scalar_timestep_unchanged(self, scheduler):
        """Existing scalar behavior must not change."""
        x = torch.randn(1, 16, 21, 64, 64)
        noise = torch.randn_like(x)
        t = scheduler.timesteps[10]
        result = scheduler.add_noise(x, noise, t)
        assert result.shape == x.shape

    def test_single_element_tensor(self, scheduler):
        """[1]-shaped timestep should work like scalar."""
        x = torch.randn(1, 16, 21, 64, 64)
        noise = torch.randn_like(x)
        t_scalar = scheduler.timesteps[10]
        t_tensor = scheduler.timesteps[10:11]
        r1 = scheduler.add_noise(x, noise, t_scalar)
        r2 = scheduler.add_noise(x, noise, t_tensor)
        assert torch.allclose(r1, r2)

    def test_batched_timestep(self, scheduler):
        """[B]-shaped timestep should apply per-sample sigma."""
        B = 4
        x = torch.randn(B, 16, 21, 64, 64)
        noise = torch.randn_like(x)
        indices = torch.tensor([5, 10, 20, 40])
        t_batch = scheduler.timesteps[indices]
        result = scheduler.add_noise(x, noise, t_batch)
        assert result.shape == x.shape
        # Verify each sample matches scalar version
        for i in range(B):
            r_single = scheduler.add_noise(
                x[i:i+1], noise[i:i+1], t_batch[i]
            )
            assert torch.allclose(result[i:i+1], r_single), f"Mismatch at sample {i}"


class TestTrainingWeight:
    def test_scalar_timestep(self, scheduler):
        t = scheduler.timesteps[10]
        w = scheduler.training_weight(t)
        assert w.dim() == 0 or w.shape == torch.Size([])

    def test_batched_timestep(self, scheduler):
        """[B]-shaped timestep should return [B]-shaped weights."""
        indices = torch.tensor([5, 10, 20, 40])
        t_batch = scheduler.timesteps[indices]
        weights = scheduler.training_weight(t_batch)
        assert weights.shape == (4,)
        # Verify each matches scalar version
        for i in range(4):
            w_single = scheduler.training_weight(t_batch[i])
            assert torch.allclose(weights[i], w_single), f"Mismatch at index {i}"
