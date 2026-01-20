"""DBA visualization helpers

DBA has multiple attention channels; recording small attention slices during
training can reveal whether semantic/geometric paths are behaving differently or
collapsing into the same patterns.
"""

from __future__ import annotations

import torch
from torch import Tensor

from instrumentation.viz import TrainingVizContext


class DecoupledAttentionViz:
    """DBA visualization hooks

    The visualization path is intentionally lightweight: it records tiny matrices
    for a handful of heads/tokens, avoiding the cost of storing full attention.
    """

    def record_attention_matrix(
        self,
        *,
        ctx: object | None,
        layer: object,
        q_cat: Tensor,
        k_cat: Tensor,
    ) -> None:
        if ctx is None or not isinstance(ctx, TrainingVizContext) or not ctx.enabled:
            return
        idx = int(getattr(layer, "_viz_index", -1))
        if idx < 0:
            return
        name = str(getattr(layer, "_viz_name", ""))
        mode = str(getattr(getattr(layer, "mode", None), "value", getattr(layer, "mode", "")))

        tq = int(min(int(ctx.max_tokens), int(q_cat.size(-2))))
        tk = int(min(int(ctx.max_tokens), int(k_cat.size(-2))))
        hh = int(min(int(ctx.max_heads), int(q_cat.size(1))))
        if tq <= 0 or tk <= 0 or hh <= 0:
            return

        qs = q_cat[:, :hh, :tq, :].float()
        ks = k_cat[:, :hh, :tk, :].float()
        logits = torch.matmul(qs, ks.transpose(-2, -1))
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

