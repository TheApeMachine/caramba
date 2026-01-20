"""Utility-Aligned Attention (UAA) training utilities.

UAA adds an auxiliary loss that aligns selected attention heads to a
counterfactual token-utility distribution estimated by a frozen teacher.

This module is intentionally trainer-facing (not model-facing) so it can be
iterated quickly in research without entangling core model code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config.train import UAAConfig
from instrumentation.viz import TrainingUAAContext


def _as_model(module: nn.Module) -> Any:
    """Access internal Model-like API without importing `model.Model` here."""
    # The repo's `model.Model` exposes: embedder, topology, _features_to_logits.
    return module


@dataclass
class UAAState:
    teacher: nn.Module
    cfg: UAAConfig

    def __post_init__(self) -> None:
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _teacher_logits_from_embeds(self, embeds: Tensor) -> Tensor:
        m = _as_model(self.teacher)
        feats = m.topology(embeds, ctx=None)  # type: ignore[call-arg]
        logits = m._features_to_logits(feats)  # type: ignore[attr-defined]
        if not isinstance(logits, Tensor):
            raise TypeError("Teacher _features_to_logits must return a Tensor.")
        return logits

    @torch.no_grad()
    def _teacher_embed(self, input_ids: Tensor) -> Tensor:
        m = _as_model(self.teacher)
        embeds = m.embedder(input_ids)  # type: ignore[call-arg]
        if not isinstance(embeds, Tensor):
            raise TypeError("Teacher embedder must return a Tensor.")
        return embeds

    def compute_loss(
        self,
        *,
        ctx: TrainingUAAContext,
        batch: Any,
        student_outputs: dict[str, Any],
        step_1: int,
    ) -> Tensor | None:
        """Compute the UAA attention-alignment loss for this microbatch.

        Returns:
          - scalar Tensor if UAA active
          - None if disabled / not scheduled / missing captures
        """
        if not bool(self.cfg.enabled) or not bool(ctx.uaa_enabled):
            return None
        every = max(1, int(self.cfg.every_steps))
        if (int(step_1) % every) != 0:
            return None

        # Captures must exist for configured layers.
        if not ctx.uaa_attn:
            return None

        input_ids = batch.get("input_ids", None)
        target_ids = batch.get("target_ids", None)
        if not isinstance(input_ids, Tensor) or not isinstance(target_ids, Tensor):
            return None
        if input_ids.ndim != 2 or target_ids.ndim != 2:
            return None
        B, T = input_ids.shape
        if target_ids.shape != (B, T):
            return None

        # Query position (only "last" supported).
        q_idx = int(T - 1)
        if int(q_idx) < 0:
            return None

        # Teacher baseline embeds/logits.
        embeds = self._teacher_embed(input_ids)
        logits_base = self._teacher_logits_from_embeds(embeds)
        if logits_base.ndim != 3 or logits_base.shape[:2] != (B, T):
            return None

        # Per-example baseline CE at query position.
        y = target_ids[:, q_idx].long()
        base_loss = F.cross_entropy(
            logits_base[:, q_idx, :].float(),
            y,
            reduction="none",
        )  # (B,)

        # Sample one token index per example from the available prefix [0..q_idx].
        # (Keeping this constant-factor cheap is the point.)
        idx = torch.randint(
            low=0,
            high=int(q_idx) + 1,
            size=(B,),
            device=input_ids.device,
            dtype=torch.long,
        )

        if str(self.cfg.counterfactual) != "embed_zero":
            raise ValueError(f"Unsupported UAA counterfactual={self.cfg.counterfactual!r}")

        embeds_cf = embeds.clone()
        embeds_cf[torch.arange(B, device=embeds_cf.device), idx, :] = 0.0
        logits_cf = self._teacher_logits_from_embeds(embeds_cf)
        cf_loss = F.cross_entropy(
            logits_cf[:, q_idx, :].float(),
            y,
            reduction="none",
        )  # (B,)

        util = torch.relu(cf_loss - base_loss)  # (B,)

        # Build target distribution q over key positions.
        eps = float(self.cfg.epsilon)
        q_raw = torch.zeros((B, T), device=input_ids.device, dtype=torch.float32)
        q_raw[:, : q_idx + 1] = eps
        q_raw[torch.arange(B, device=q_raw.device), idx] += util.float()
        q = q_raw / torch.clamp(q_raw.sum(dim=-1, keepdim=True), min=1e-12)  # (B,T)

        # Align each captured (layer, heads) distribution to q via KL(q || a).
        losses: list[Tensor] = []
        for layer_idx, a in ctx.uaa_attn.items():
            if not isinstance(a, Tensor):
                continue
            if a.ndim != 3:
                continue
            if a.shape[0] != B or a.shape[2] != T:
                continue

            # a: (B, H_sel, T)
            a_safe = a.float().clamp_min(1e-9)
            a_norm = a_safe / torch.clamp(a_safe.sum(dim=-1, keepdim=True), min=1e-12)

            q_safe = q.clamp_min(1e-9)
            q_norm = q_safe / torch.clamp(q_safe.sum(dim=-1, keepdim=True), min=1e-12)

            # Broadcast q to heads: (B,1,T) -> (B,H,T)
            qh = q_norm.unsqueeze(1).expand_as(a_norm)
            kl = (qh * (qh.log() - a_norm.log())).sum(dim=-1)  # (B,H)
            losses.append(kl.mean())

        if not losses:
            return None

        att_kl = torch.stack(losses).mean()

        # Expose unweighted KL for logging/inspection (trainer will pick it up).
        student_outputs["uaa/attn_kl"] = att_kl.detach()
        return att_kl * float(self.cfg.lambda_att)

