from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from diffusion.samplers import DdimSampler, DdpmSampler, GuidanceConfig

VOCAB_SIZE = 8


class DummyDiffusionModel(nn.Module):
    """Minimal diffusion model for sampler tests."""

    def forward(  # type: ignore[override]
        self,
        *,
        noisy_emb: Tensor,
        t: Tensor,
        target_pad_mask: Tensor | None,
        self_cond: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        _ = (t, target_pad_mask, self_cond, prompt_emb, prompt_pad_mask)
        pred_eps = torch.zeros_like(noisy_emb)
        x0 = torch.zeros_like(noisy_emb)
        logits = torch.zeros(noisy_emb.shape[0], noisy_emb.shape[1], VOCAB_SIZE, device=noisy_emb.device)
        return pred_eps, x0, logits


class TestSamplers:
    """Tests for DDPM/DDIM samplers."""

    def test_ddim_step_indices_are_unique_and_sorted(self) -> None:
        device = torch.device("cpu")
        model: Any = DummyDiffusionModel()
        alpha_bar = torch.linspace(1.0, 0.1, 10, device=device)
        sampler = DdimSampler(model=model, alpha_bar=alpha_bar, timesteps=10, device=device, hidden_size=4, steps=50, eta=0.0)
        idx = sampler.stepIndices()
        assert int(idx.numel()) >= 2
        assert torch.all(idx[1:] >= idx[:-1])
        assert int(torch.unique(idx).numel()) == int(idx.numel())

    def test_ddpm_and_ddim_return_token_ids(self) -> None:
        device = torch.device("cpu")
        model: Any = DummyDiffusionModel()
        alpha_bar = torch.linspace(1.0, 0.1, VOCAB_SIZE, device=device)
        emb_w = torch.randn(VOCAB_SIZE, 4, device=device)
        cfg = GuidanceConfig(guidance_scale=1.0)

        ddpm = DdpmSampler(model=model, alpha_bar=alpha_bar, timesteps=VOCAB_SIZE, device=device, hidden_size=4)
        out_ddpm = ddpm.sample(batch_size=2, seq_len=6, target_pad_mask=None, prompt_emb=None, prompt_pad_mask=None, cfg=cfg, embedding_weight=emb_w)
        assert tuple(out_ddpm.shape) == (2, 6)
        assert torch.all(out_ddpm >= 0)
        assert torch.all(out_ddpm < emb_w.shape[0])

        ddim = DdimSampler(model=model, alpha_bar=alpha_bar, timesteps=VOCAB_SIZE, device=device, hidden_size=4, steps=4, eta=0.0)
        out_ddim = ddim.sample(batch_size=2, seq_len=6, target_pad_mask=None, prompt_emb=None, prompt_pad_mask=None, cfg=cfg, embedding_weight=emb_w)
        assert tuple(out_ddim.shape) == (2, 6)
        assert torch.all(out_ddim >= 0)
        assert torch.all(out_ddim < emb_w.shape[0])

