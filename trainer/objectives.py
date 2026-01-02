"""Objective implementations.

Objectives compute a *scalar* loss tensor from (batch, outputs) using a strict,
dictionary-based protocol:

- batch:   dict[str, Tensor]
- outputs: dict[str, Tensor]

The particular tensor keys are configured via the manifest so trainers stay
completely model/task agnostic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from caramba.runtime.tensordict_utils import TensorDictBase

TensorDict = TensorDictBase
MetricDict = dict[str, float]


def _require_tensor(d: Mapping[str, Any], key: str, *, where: str) -> Tensor:
    if key not in d:
        raise KeyError(f"Missing key {key!r} in {where}.")
    v = d[key]
    if not isinstance(v, Tensor):
        raise TypeError(f"Expected {where}[{key!r}] to be a Tensor, got {type(v).__name__}")
    return v


class KeyedMSEObjective:
    """Mean squared error objective for regression-like tasks."""

    def __init__(self, *, pred_key: str = "pred", target_key: str = "targets") -> None:
        self.pred_key = str(pred_key)
        self.target_key = str(target_key)

    def loss(self, *, outputs: TensorDict, batch: TensorDict) -> Tensor:
        pred = _require_tensor(outputs, self.pred_key, where="outputs")
        tgt = _require_tensor(batch, self.target_key, where="batch")
        return torch.mean((pred - tgt) ** 2)

    def metrics(self, *, outputs: TensorDict, batch: TensorDict, loss: Tensor) -> MetricDict:
        _ = outputs
        _ = batch
        return {"mse": float(loss.detach())}


class KeyedCrossEntropyObjective:
    """Cross-entropy classification objective.

    Expects:
    - outputs[logits_key] shape (B, C) or (B, ..., C)
    - batch[labels_key] shape (B,) or (B, ...)
    """

    def __init__(
        self,
        *,
        logits_key: str = "logits",
        labels_key: str = "labels",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        self.logits_key = str(logits_key)
        self.labels_key = str(labels_key)
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

    def loss(self, *, outputs: TensorDict, batch: TensorDict) -> Tensor:
        logits = _require_tensor(outputs, self.logits_key, where="outputs")
        labels = _require_tensor(batch, self.labels_key, where="batch").long()
        return F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

    def metrics(self, *, outputs: TensorDict, batch: TensorDict, loss: Tensor) -> MetricDict:
        _ = outputs
        _ = batch
        return {"ce_loss": float(loss.detach())}


class NextTokenCrossEntropyObjective(KeyedCrossEntropyObjective):
    """Legacy name for a keyed cross entropy objective (LM next-token by default)."""

    def __init__(
        self,
        *,
        logits_key: str = "logits",
        target_key: str = "target_ids",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(
            logits_key=logits_key,
            labels_key=target_key,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        # Backwards-compatible alias.
        self.target_key = self.labels_key


class MosaicNextTokenWithAuxObjective(NextTokenCrossEntropyObjective):
    """Next-token CE + auxiliary curriculum losses for MOSAIC memory control.

    Expected (optional) batch keys:
    - mosaic_teacher_write_gate: (B,T) float in {0,1} or -1 to ignore
    - mosaic_teacher_read_bucket: (B,T) or (B,T,H) int64 bucket index, -1 to ignore
    - mosaic_teacher_write_bucket: (B,T) or (B,T,H) int64 bucket index, -1 to ignore

    Expected (optional) output keys (provided by MOSAIC layers via ctx):
    - mosaic_write_gate_logits: (B,T) logits
    - mosaic_read_bit_logits: (B,T,H,BITS) logits (before sign)
    - mosaic_write_bit_logits: (B,T,H,BITS) logits (before sign)
    """

    def __init__(
        self,
        *,
        logits_key: str = "logits",
        target_key: str = "target_ids",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        aux_gate_weight: float = 0.1,
        aux_bits_weight: float = 0.1,
        aux_utility_weight: float = 0.1,
        aux_contrastive_weight: float = 0.1,
    ) -> None:
        super().__init__(
            logits_key=logits_key,
            target_key=target_key,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        self.aux_gate_weight = float(aux_gate_weight)
        self.aux_bits_weight = float(aux_bits_weight)
        self.aux_utility_weight = float(aux_utility_weight)
        self.aux_contrastive_weight = float(aux_contrastive_weight)

    @staticmethod
    def _bucket_to_bits(bucket: Tensor, *, n_bits: int) -> Tensor:
        """Convert bucket indices to {0,1} bits along a last dimension."""
        # bucket: (...,) int64
        b = bucket.long().clamp_min(0)
        shifts = torch.arange(int(n_bits), device=b.device, dtype=torch.long)
        bits = ((b.unsqueeze(-1) >> shifts) & 1).to(dtype=torch.float32)
        return bits

    def loss(self, *, outputs: TensorDict, batch: TensorDict) -> Tensor:
        base = super().loss(outputs=outputs, batch=batch)
        loss = base

        # Gate imitation (optional).
        tg = batch.get("mosaic_teacher_write_gate", None) if isinstance(batch, dict) else None
        pg = outputs.get("mosaic_write_gate_logits", None) if isinstance(outputs, dict) else None
        if (
            self.aux_gate_weight > 0.0
            and isinstance(tg, Tensor)
            and isinstance(pg, Tensor)
            and tg.shape == pg.shape
        ):
            # Mask ignored entries.
            mask = tg >= 0
            if bool(mask.any().item()):
                gate_loss = F.binary_cross_entropy_with_logits(
                    pg[mask].float(),
                    tg[mask].float(),
                )
                loss = loss + float(self.aux_gate_weight) * gate_loss

        # Address imitation via bit logits (optional).
        # We supervise bits rather than bucket softmax to avoid O(B) outputs.
        tr = batch.get("mosaic_teacher_read_bucket", None) if isinstance(batch, dict) else None
        tw = batch.get("mosaic_teacher_write_bucket", None) if isinstance(batch, dict) else None
        pr = outputs.get("mosaic_read_bit_logits", None) if isinstance(outputs, dict) else None
        pw = outputs.get("mosaic_write_bit_logits", None) if isinstance(outputs, dict) else None

        def _bits_loss(t_bucket: Tensor, p_bits: Tensor) -> Tensor | None:
            # t_bucket: (B,T) or (B,T,H)
            # p_bits: (B,T,H,BITS)
            if p_bits.ndim != 4:
                return None
            B, T, H, n_bits = p_bits.shape
            tb = t_bucket
            if tb.ndim == 2:
                tb = tb.unsqueeze(-1).expand(B, T, H)
            if tb.ndim != 3 or tb.shape[:3] != (B, T, H):
                return None
            mask = tb >= 0
            if not bool(mask.any().item()):
                return None
            tgt_bits = self._bucket_to_bits(tb, n_bits=int(n_bits))
            # Apply mask to bits.
            m = mask.unsqueeze(-1).expand_as(tgt_bits)
            return F.binary_cross_entropy_with_logits(
                p_bits[m].float(),
                tgt_bits[m].float(),
            )

        if self.aux_bits_weight > 0.0 and isinstance(pr, Tensor) and isinstance(tr, Tensor):
            bl = _bits_loss(tr, pr)
            if bl is not None:
                loss = loss + float(self.aux_bits_weight) * bl
        if self.aux_bits_weight > 0.0 and isinstance(pw, Tensor) and isinstance(tw, Tensor):
            bl2 = _bits_loss(tw, pw)
            if bl2 is not None:
                loss = loss + float(self.aux_bits_weight) * bl2

        # VQ router imitation (optional): CE over per-group code logits.
        vqr = outputs.get("mosaic_vq_read_logits", None) if isinstance(outputs, dict) else None
        vqw = outputs.get("mosaic_vq_write_logits", None) if isinstance(outputs, dict) else None

        def _decode_codes(bucket: Tensor, *, K: int, G: int) -> Tensor:
            # bucket: (B,T,H)
            b = bucket.long().clamp_min(0)
            codes = []
            for g in range(int(G)):
                codes.append(((b // (K**g)) % K).unsqueeze(-1))
            return torch.cat(codes, dim=-1)  # (B,T,H,G)

        def _vq_ce(t_bucket: Tensor, p_logits: Tensor) -> Tensor | None:
            # t_bucket: (B,T) or (B,T,H)
            # p_logits: (B,T,H,G,K)
            if p_logits.ndim != 5:
                return None
            B, T, H, G, K = p_logits.shape
            tb = t_bucket
            if tb.ndim == 2:
                tb = tb.unsqueeze(-1).expand(B, T, H)
            if tb.ndim != 3 or tb.shape != (B, T, H):
                return None
            mask = tb >= 0
            if not bool(mask.any().item()):
                return None
            tgt_codes = _decode_codes(tb, K=int(K), G=int(G))  # (B,T,H,G)
            # Flatten.
            p = p_logits.reshape(B * T * H * G, K)
            tgt = tgt_codes.reshape(B * T * H * G)
            m = mask.unsqueeze(-1).expand(B, T, H, G).reshape(B * T * H * G)
            if not bool(m.any().item()):
                return None
            return F.cross_entropy(p[m], tgt[m])

        if self.aux_bits_weight > 0.0 and isinstance(vqr, Tensor) and isinstance(tr, Tensor):
            vql = _vq_ce(tr, vqr)
            if vql is not None:
                loss = loss + float(self.aux_bits_weight) * vql
        if self.aux_bits_weight > 0.0 and isinstance(vqw, Tensor) and isinstance(tw, Tensor):
            vql2 = _vq_ce(tw, vqw)
            if vql2 is not None:
                loss = loss + float(self.aux_bits_weight) * vql2

        # Utility prediction imitation (optional).
        tu = batch.get("mosaic_teacher_write_utility", None) if isinstance(batch, dict) else None
        pu = outputs.get("mosaic_write_utility_logits", None) if isinstance(outputs, dict) else None
        if (
            self.aux_utility_weight > 0.0
            and isinstance(tu, Tensor)
            and isinstance(pu, Tensor)
            and tu.shape == pu.shape
        ):
            mask = tu >= 0
            if bool(mask.any().item()):
                ul = F.binary_cross_entropy_with_logits(pu[mask].float(), tu[mask].float())
                loss = loss + float(self.aux_utility_weight) * ul

        # Contrastive auxiliary from the model (optional scalar).
        cl = outputs.get("mosaic_contrastive_loss", None) if isinstance(outputs, dict) else None
        if self.aux_contrastive_weight > 0.0 and isinstance(cl, Tensor):
            # Expect scalar tensor; if not, reduce.
            cval = cl.float().mean()
            loss = loss + float(self.aux_contrastive_weight) * cval

        return loss

    def metrics(self, *, outputs: TensorDict, batch: TensorDict, loss: Tensor) -> MetricDict:
        """Expose per-component losses for debugging new memory behaviors."""
        m: MetricDict = {
            "loss_total": float(loss.detach()),
        }
        try:
            base = super().loss(outputs=outputs, batch=batch)
            m["lm_ce"] = float(base.detach())
        except Exception:
            m["lm_ce"] = float("nan")

        tg = batch.get("mosaic_teacher_write_gate", None) if isinstance(batch, dict) else None
        pg = outputs.get("mosaic_write_gate_logits", None) if isinstance(outputs, dict) else None
        tr = batch.get("mosaic_teacher_read_bucket", None) if isinstance(batch, dict) else None
        tw = batch.get("mosaic_teacher_write_bucket", None) if isinstance(batch, dict) else None
        pr = outputs.get("mosaic_read_bit_logits", None) if isinstance(outputs, dict) else None
        pw = outputs.get("mosaic_write_bit_logits", None) if isinstance(outputs, dict) else None
        vqr = outputs.get("mosaic_vq_read_logits", None) if isinstance(outputs, dict) else None
        vqw = outputs.get("mosaic_vq_write_logits", None) if isinstance(outputs, dict) else None
        tu = batch.get("mosaic_teacher_write_utility", None) if isinstance(batch, dict) else None
        pu = outputs.get("mosaic_write_utility_logits", None) if isinstance(outputs, dict) else None
        cl = outputs.get("mosaic_contrastive_loss", None) if isinstance(outputs, dict) else None

        # Gate imitation (BCE).
        gate_loss = None
        if isinstance(tg, Tensor) and isinstance(pg, Tensor) and tg.shape == pg.shape:
            mask = tg >= 0
            if bool(mask.any().item()):
                try:
                    gate_loss = F.binary_cross_entropy_with_logits(pg[mask].float(), tg[mask].float())
                except Exception:
                    gate_loss = None
        if gate_loss is not None:
            m["aux_gate_bce"] = float(gate_loss.detach())
            m["aux_gate_weighted"] = float((gate_loss * float(self.aux_gate_weight)).detach())

        def _bits_loss(t_bucket: Tensor, p_bits: Tensor) -> Tensor | None:
            if p_bits.ndim != 4:
                return None
            B, T, H, n_bits = p_bits.shape
            tb = t_bucket
            if tb.ndim == 2:
                tb = tb.unsqueeze(-1).expand(B, T, H)
            if tb.ndim != 3 or tb.shape[:3] != (B, T, H):
                return None
            mask = tb >= 0
            if not bool(mask.any().item()):
                return None
            tgt_bits = self._bucket_to_bits(tb, n_bits=int(n_bits))
            msk = mask.unsqueeze(-1).expand_as(tgt_bits)
            return F.binary_cross_entropy_with_logits(p_bits[msk].float(), tgt_bits[msk].float())

        bits_r = None
        bits_w = None
        if isinstance(tr, Tensor) and isinstance(pr, Tensor):
            try:
                bits_r = _bits_loss(tr, pr)
            except Exception:
                bits_r = None
        if isinstance(tw, Tensor) and isinstance(pw, Tensor):
            try:
                bits_w = _bits_loss(tw, pw)
            except Exception:
                bits_w = None
        if bits_r is not None:
            m["aux_bits_bce_read"] = float(bits_r.detach())
        if bits_w is not None:
            m["aux_bits_bce_write"] = float(bits_w.detach())
        if bits_r is not None or bits_w is not None:
            val = (bits_r if bits_r is not None else 0.0) + (bits_w if bits_w is not None else 0.0)
            if isinstance(val, Tensor):
                m["aux_bits_weighted"] = float((val * float(self.aux_bits_weight)).detach())

        def _decode_codes(bucket: Tensor, *, K: int, G: int) -> Tensor:
            b = bucket.long().clamp_min(0)
            codes = []
            for g in range(int(G)):
                codes.append(((b // (K**g)) % K).unsqueeze(-1))
            return torch.cat(codes, dim=-1)

        def _vq_ce(t_bucket: Tensor, p_logits: Tensor) -> Tensor | None:
            if p_logits.ndim != 5:
                return None
            B, T, H, G, K = p_logits.shape
            tb = t_bucket
            if tb.ndim == 2:
                tb = tb.unsqueeze(-1).expand(B, T, H)
            if tb.ndim != 3 or tb.shape != (B, T, H):
                return None
            mask = tb >= 0
            if not bool(mask.any().item()):
                return None
            tgt_codes = _decode_codes(tb, K=int(K), G=int(G))  # (B,T,H,G)
            p = p_logits.reshape(B * T * H * G, K)
            tgt = tgt_codes.reshape(B * T * H * G)
            mm = mask.unsqueeze(-1).expand(B, T, H, G).reshape(B * T * H * G)
            if not bool(mm.any().item()):
                return None
            return F.cross_entropy(p[mm], tgt[mm])

        vq_r = None
        vq_w = None
        if isinstance(tr, Tensor) and isinstance(vqr, Tensor):
            try:
                vq_r = _vq_ce(tr, vqr)
            except Exception:
                vq_r = None
        if isinstance(tw, Tensor) and isinstance(vqw, Tensor):
            try:
                vq_w = _vq_ce(tw, vqw)
            except Exception:
                vq_w = None
        if vq_r is not None:
            m["aux_vq_ce_read"] = float(vq_r.detach())
        if vq_w is not None:
            m["aux_vq_ce_write"] = float(vq_w.detach())
        if vq_r is not None or vq_w is not None:
            vv = (vq_r if vq_r is not None else 0.0) + (vq_w if vq_w is not None else 0.0)
            if isinstance(vv, Tensor):
                m["aux_vq_weighted"] = float((vv * float(self.aux_bits_weight)).detach())

        util_loss = None
        if isinstance(tu, Tensor) and isinstance(pu, Tensor) and tu.shape == pu.shape:
            mask = tu >= 0
            if bool(mask.any().item()):
                try:
                    util_loss = F.binary_cross_entropy_with_logits(pu[mask].float(), tu[mask].float())
                except Exception:
                    util_loss = None
        if util_loss is not None:
            m["aux_utility_bce"] = float(util_loss.detach())
            m["aux_utility_weighted"] = float((util_loss * float(self.aux_utility_weight)).detach())

        if isinstance(cl, Tensor):
            try:
                cval = cl.float().mean()
                m["aux_contrastive"] = float(cval.detach())
                m["aux_contrastive_weighted"] = float((cval * float(self.aux_contrastive_weight)).detach())
            except Exception:
                pass

        return m

