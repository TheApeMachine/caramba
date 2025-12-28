"""Toy diffusion denoiser system on vector data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


def _sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / (half - 1)))
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


@dataclass
class DiffusionDenoiserSystem:
    """Predict noise eps from (x_t, t) using an MLP."""

    d_in: int
    d_hidden: int = 256
    time_dim: int = 64
    n_layers: int = 2

    def __post_init__(self) -> None:
        layers: list[nn.Module] = []
        din = int(self.d_in) + int(self.time_dim)
        for _ in range(int(self.n_layers)):
            layers.append(nn.Linear(din, int(self.d_hidden)))
            layers.append(nn.SiLU())
            din = int(self.d_hidden)
        layers.append(nn.Linear(din, int(self.d_in)))
        self.module = nn.Sequential(*layers)

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "DiffusionDenoiserSystem":
        self.module = self.module.to(device=device, dtype=dtype)
        return self

    def forward(self, batch: dict[str, Any], *, ctx: object | None = None) -> dict[str, Any]:
        _ = ctx
        x_t: Tensor = batch["x_t"]
        t: Tensor = batch["t"].view(-1)
        t_emb = _sinusoidal_time_embedding(t, int(self.time_dim)).to(dtype=x_t.dtype)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        h = torch.cat([x_t, t_emb], dim=-1)
        pred_eps = self.module(h)
        return {"pred_eps": pred_eps}

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.module.load_state_dict(state)

