"""Diffusion code generation system

Implements a diffusion-on-embeddings transformer that supports:
- conditional generation via prompt cross-attention
- classifier-free guidance by running conditional + unconditional passes
- self-conditioning by feeding previous x0 predictions into the denoiser
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pathlib import Path

from data.tokenizers.hf_json import HfJsonTokenizer


class TimestepEmbedding(nn.Module):
    """Timestep embedding

    Converts an integer diffusion timestep into a sinusoidal embedding that can
    be added to token embeddings to condition the denoiser on diffusion time.
    """

    def __init__(self, *, dim: int, max_period: int = 10_000) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_period = int(max_period)

    def forward(self, t: Tensor) -> Tensor:
        half = int(self.dim) // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(float(self.max_period), device=t.device))
            * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32)
            / max(1, half)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if int(self.dim) % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class PositionalEmbedding(nn.Module):
    """Learned positional embedding

    Uses an embedding table over positions to provide token position information.
    """

    def __init__(self, *, max_len: int, dim: int) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.dim = int(dim)
        self.table = nn.Embedding(self.max_len, self.dim)

    def forward(self, x: Tensor) -> Tensor:
        """Return learned embeddings for each position in `x`.

        Note: this module uses `x` **only for its shape/device** (expects `(B, L, H)`),
        and returns a `(B, L, dim)` tensor of learned per-position embeddings.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to be (B,L,H), got shape={tuple(x.shape)}")
        batch, length, _hidden = x.shape
        if int(length) > int(self.max_len):
            raise ValueError(
                "Sequence length exceeds max_len for positional embedding. "
                f"length={int(length)}, max_len={int(self.max_len)}"
            )
        pos = torch.arange(length, device=x.device).unsqueeze(0).expand(batch, length)
        return self.table(pos)


class DiffusionTransformer(nn.Module):
    """Prompt-conditioned diffusion denoiser

    Decoder-only denoiser that uses cross-attention into the prompt "memory"
    while denoising a noisy target embedding sequence.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        max_len: int,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_id)
        self.positional = PositionalEmbedding(max_len=self.max_len, dim=self.hidden_size)
        self.timestep = TimestepEmbedding(dim=self.hidden_size)
        self.selfConditionProjection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(num_layers))
        self.head = nn.Linear(self.hidden_size, self.hidden_size * 2)

    def forward(
        self,
        *,
        noisy_emb: Tensor,
        t: Tensor,
        target_pad_mask: Tensor | None,
        self_cond: Tensor | None,
        prompt_emb: Tensor | None,
        prompt_pad_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self_cond is None:
            self_cond = torch.zeros_like(noisy_emb)

        target = noisy_emb
        target = target + self.selfConditionProjection(self_cond)
        target = target + self.positional(noisy_emb)
        target = target + self.timestep(t).unsqueeze(1).to(dtype=target.dtype)

        memory, memory_mask = self.prepareMemory(target=target, prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask)

        out = self.decoder(
            tgt=target,
            memory=memory,
            tgt_key_padding_mask=target_pad_mask,
            memory_key_padding_mask=memory_mask,
        )
        pred_eps, x0_pred = self.head(out).chunk(2, dim=-1)
        logits = F.linear(x0_pred, self.embedding.weight)
        return pred_eps, x0_pred, logits

    def prepareMemory(
        self, *, target: Tensor, prompt_emb: Tensor | None, prompt_pad_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if prompt_emb is None:
            memory = torch.zeros_like(target)
            memory_mask = torch.ones((target.size(0), target.size(1)), device=target.device, dtype=torch.bool)
            return memory, memory_mask

        memory = prompt_emb + self.positional(prompt_emb)
        if prompt_pad_mask is None:
            raise ValueError("prompt_pad_mask is required when prompt_emb is provided.")
        return memory, prompt_pad_mask


@dataclass
class DiffusionCodegenSystem:
    """Diffusion code generation system

    Wraps the prompt-conditioned diffusion transformer behind the system
    protocol used by Caramba trainers (forward/to/state_dict/load_state_dict).
    """

    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    dim_feedforward: int
    max_len: int
    pad_id: int | None = None
    tokenizer_file: str | None = None
    pad_token: str = "<pad>"

    def __post_init__(self) -> None:
        pad_id = self.pad_id
        if pad_id is None:
            pad_id = self.resolvePadId()
        self.module = DiffusionTransformer(
            vocab_size=int(self.vocab_size),
            hidden_size=int(self.hidden_size),
            num_layers=int(self.num_layers),
            num_heads=int(self.num_heads),
            dim_feedforward=int(self.dim_feedforward),
            max_len=int(self.max_len),
            pad_id=int(pad_id),
        )

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "DiffusionCodegenSystem":
        self.module = self.module.to(device=device, dtype=dtype)
        return self

    def train(self, mode: bool = True) -> None:
        self.module.train(mode)

    def eval(self) -> None:
        self.module.eval()

    def forward(self, batch: dict[str, Any], *, ctx: object | None = None) -> dict[str, Any]:
        _ = ctx
        noisy_emb = self.requireTensor(batch=batch, key="noisy_emb")
        t = self.requireTensor(batch=batch, key="t")
        target_pad_mask = batch.get("target_pad_mask", None)
        self_cond = batch.get("self_cond", None)
        prompt_emb = batch.get("prompt_emb", None)
        prompt_pad_mask = batch.get("prompt_pad_mask", None)

        pred_eps, x0_pred, logits = self.module(
            noisy_emb=noisy_emb,
            t=t,
            target_pad_mask=target_pad_mask,
            self_cond=self_cond,
            prompt_emb=prompt_emb,
            prompt_pad_mask=prompt_pad_mask,
        )
        return {"pred_eps": pred_eps, "x0_pred": x0_pred, "logits": logits}

    def requireTensor(self, *, batch: dict[str, Any], key: str) -> Tensor:
        if key not in batch:
            raise KeyError(f"Missing batch key {key!r}")
        x = batch[key]
        if not isinstance(x, Tensor):
            raise TypeError(f"Expected batch[{key!r}] to be a Tensor, got {type(x).__name__}")
        return x

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.module.load_state_dict(state)

    def resolvePadId(self) -> int:
        """Resolve pad_id from a tokenizer file."""

        if not self.tokenizer_file:
            raise ValueError(
                "pad_id was not provided and tokenizer_file is missing. "
                "Provide system.config.pad_id or system.config.tokenizer_file."
            )

        path = Path(str(self.tokenizer_file))
        if not path.exists():
            raise FileNotFoundError(
                f"tokenizer_file not found: {path}. "
                "Train the tokenizer first or set system.config.tokenizer_file."
            )
        tokenizer = HfJsonTokenizer.from_file(tokenizer_file=str(path))
        pad_id = tokenizer.token_to_id(str(self.pad_token))
        if pad_id is None:
            raise ValueError(
                f"Tokenizer does not define pad_token={self.pad_token!r}. "
                "Ensure the tokenizer was trained with this special token."
            )
        return int(pad_id)
