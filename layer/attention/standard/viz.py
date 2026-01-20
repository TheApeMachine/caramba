"""Attention visualization helpers

Attention is hard to debug from losses alone; these utilities record small,
cheap-to-compute summaries (like tiny attention matrices and entropy) so you can
see what heads are doing during training.
"""

from __future__ import annotations

import torch
from torch import Tensor

from instrumentation.viz import TrainingVizContext


class AttentionViz:
    """Attention visualization hooks

    SDPA kernels are optimized for speed and do not expose full attention
    weights, so we recompute a small slice for instrumentation only.
    """

    def record_attention_matrix(
        self,
        *,
        ctx: object | None,
        layer: object,
        q: Tensor,
        k: Tensor,
        scale: float,
        causal: bool,
    ) -> None:
        if ctx is None or not isinstance(ctx, TrainingVizContext) or not ctx.enabled:
            return
        idx = int(getattr(layer, "_viz_index", -1))
        if idx < 0:
            return

        name = str(getattr(layer, "_viz_name", ""))
        mode = str(getattr(getattr(layer, "mode", None), "value", getattr(layer, "mode", "")))
        tq = int(min(int(ctx.max_tokens), int(q.size(2))))
        tk = int(min(int(ctx.max_tokens), int(k.size(2))))
        hh = int(min(int(ctx.max_heads), int(q.size(1))))
        if tq <= 0 or tk <= 0 or hh <= 0:
            return

        qs = q[:, :hh, :tq, :].float()
        ks = k[:, :hh, :tk, :].float()
        logits = torch.matmul(qs, ks.transpose(-2, -1)) * float(scale)
        if bool(causal):
            causal_mask = torch.tril(torch.ones((tq, tk), device=logits.device, dtype=torch.bool))
            logits = logits.masked_fill(~causal_mask.view(1, 1, tq, tk), float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        eps = 1e-9
        ent = -(probs * (probs + eps).log()).sum(dim=-1).mean(dim=-1)  # (B,H)
        ent_list = [float(x) for x in ent[0].detach().cpu().tolist()]
        mats = [probs[0, h, :, :].detach() for h in range(int(hh))]
        ctx.record_attention_matrix(
            idx=idx,
            name=name,
            mode=mode,
            n_heads=int(getattr(layer, "n_heads", hh)),
            matrices=mats,
            entropies=ent_list,
        )

    def record_activation_sample(self, *, ctx: object | None, layer: object, y: Tensor) -> None:
        if ctx is None or not isinstance(ctx, TrainingVizContext) or not ctx.enabled:
            return
        idx = int(getattr(layer, "_viz_index", -1))
        if idx < 0:
            return
        name = str(getattr(layer, "_viz_name", ""))
        mode = str(getattr(getattr(layer, "mode", None), "value", getattr(layer, "mode", "")))
        raw_n_heads = getattr(layer, "n_heads", None)
        try:
            n_heads = int(raw_n_heads) if raw_n_heads is not None else None
        except Exception:
            n_heads = None
        if not n_heads:
            n_heads = None
        ctx.record_activation_sample(
            idx=idx,
            name=name,
            mode=mode,
            n_heads=n_heads,
            y=y,
        )

