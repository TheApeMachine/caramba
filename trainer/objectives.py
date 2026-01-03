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


def _maybe_get(obj: object, key: str) -> object | None:
    """Best-effort Mapping/TensorDict getter.

    `TensorDictBase` is dict-like but not necessarily a `dict`, so we can't rely on
    `isinstance(x, dict)` checks when reading optional keys.
    """
    if obj is None:
        return None
    try:
        getter = getattr(obj, "get", None)
        if callable(getter):
            return getter(key, None)
    except Exception:
        pass
    try:
        if key in obj:  # type: ignore[operator]
            return obj[key]  # type: ignore[index]
    except Exception:
        pass
    return None


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
    - mosaic_teacher_opcode: (B,T) int64 opcode id, -1 to ignore
    - mosaic_teacher_commitment_delta: (B,T) int64 in {-1,0,1} or -100 to ignore
    - mosaic_teacher_reg_write_gate: (B,T) float in {0,1} or -1 to ignore
    - mosaic_teacher_reg_sel: (B,T) int64 register slot id, -1 to ignore

    Expected (optional) output keys (provided by MOSAIC layers via ctx):
    - mosaic_write_gate_logits: (B,T) logits
    - mosaic_read_bit_logits: (B,T,H,BITS) logits (before sign)
    - mosaic_write_bit_logits: (B,T,H,BITS) logits (before sign)
    - mosaic_opcode_logits: (B,T,OP) logits
    - mosaic_commitment_logits: (B,T,3) logits [close, neutral, open]
    - mosaic_reg_write_gate_logits: (B,T) logits
    - mosaic_reg_sel_logits: (B,T,R) logits
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
        aux_opcode_weight: float = 0.1,
        aux_commitment_weight: float = 0.0,
        aux_reg_gate_weight: float = 0.0,
        aux_reg_sel_weight: float = 0.0,
        aux_state_decay_weight: float = 0.0,
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
        self.aux_opcode_weight = float(aux_opcode_weight)
        self.aux_commitment_weight = float(aux_commitment_weight)
        self.aux_reg_gate_weight = float(aux_reg_gate_weight)
        self.aux_reg_sel_weight = float(aux_reg_sel_weight)
        self.aux_state_decay_weight = float(aux_state_decay_weight)

    @staticmethod
    def _bucket_to_bits(bucket: Tensor, *, n_bits: int) -> Tensor:
        """Convert bucket indices to {0,1} bits along a last dimension."""
        # bucket: (...,) int64
        b = bucket.long().clamp_min(0)
        shifts = torch.arange(int(n_bits), device=b.device, dtype=torch.long)
        bits = ((b.unsqueeze(-1) >> shifts) & 1).to(dtype=torch.float32)
        return bits

    @staticmethod
    def _decode_codes(bucket: Tensor, *, K: int, G: int) -> Tensor:
        # bucket: (B,T,H)
        b = bucket.long().clamp_min(0)
        codes = []
        for g in range(int(G)):
            codes.append(((b // (K**g)) % K).unsqueeze(-1))
        return torch.cat(codes, dim=-1)  # (B,T,H,G)

    def _bits_loss(self, t_bucket: Tensor, p_bits: Tensor) -> Tensor | None:
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

    def _vq_ce(self, t_bucket: Tensor, p_logits: Tensor) -> Tensor | None:
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
        tgt_codes = self._decode_codes(tb, K=int(K), G=int(G))  # (B,T,H,G)
        # Flatten.
        p = p_logits.reshape(B * T * H * G, K)
        tgt = tgt_codes.reshape(B * T * H * G)
        m = mask.unsqueeze(-1).expand(B, T, H, G).reshape(B * T * H * G)
        if not bool(m.any().item()):
            return None
        return F.cross_entropy(p[m], tgt[m])

    def loss(self, *, outputs: TensorDict, batch: TensorDict) -> Tensor:
        base = super().loss(outputs=outputs, batch=batch)
        loss = base

        # Gate imitation (optional).
        tg = _maybe_get(batch, "mosaic_teacher_write_gate")
        pg = _maybe_get(outputs, "mosaic_write_gate_logits")
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
        tr = _maybe_get(batch, "mosaic_teacher_read_bucket")
        tw = _maybe_get(batch, "mosaic_teacher_write_bucket")
        pr = _maybe_get(outputs, "mosaic_read_bit_logits")
        pw = _maybe_get(outputs, "mosaic_write_bit_logits")

        if self.aux_bits_weight > 0.0 and isinstance(pr, Tensor) and isinstance(tr, Tensor):
            bl = self._bits_loss(tr, pr)
            if bl is not None:
                loss = loss + float(self.aux_bits_weight) * bl
        if self.aux_bits_weight > 0.0 and isinstance(pw, Tensor) and isinstance(tw, Tensor):
            bl2 = self._bits_loss(tw, pw)
            if bl2 is not None:
                loss = loss + float(self.aux_bits_weight) * bl2

        # VQ router imitation (optional): CE over per-group code logits.
        vqr = _maybe_get(outputs, "mosaic_vq_read_logits")
        vqw = _maybe_get(outputs, "mosaic_vq_write_logits")

        if self.aux_bits_weight > 0.0 and isinstance(vqr, Tensor) and isinstance(tr, Tensor):
            vql = self._vq_ce(tr, vqr)
            if vql is not None:
                loss = loss + float(self.aux_bits_weight) * vql
        if self.aux_bits_weight > 0.0 and isinstance(vqw, Tensor) and isinstance(tw, Tensor):
            vql2 = self._vq_ce(tw, vqw)
            if vql2 is not None:
                loss = loss + float(self.aux_bits_weight) * vql2

        # Utility prediction imitation (optional).
        tu = _maybe_get(batch, "mosaic_teacher_write_utility")
        pu = _maybe_get(outputs, "mosaic_write_utility_logits")
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

        # Opcode imitation (optional).
        to = _maybe_get(batch, "mosaic_teacher_opcode")
        po = _maybe_get(outputs, "mosaic_opcode_logits")
        if (
            self.aux_opcode_weight > 0.0
            and isinstance(to, Tensor)
            and isinstance(po, Tensor)
            and po.ndim == 3
            and to.shape == po.shape[:2]
        ):
            # CE over the opcode vocab, with -1 as ignore.
            ce = F.cross_entropy(
                po.float().view(-1, int(po.size(-1))),
                to.long().view(-1),
                ignore_index=-1,
            )
            loss = loss + float(self.aux_opcode_weight) * ce

        # Commitment delta supervision (optional).
        tc = _maybe_get(batch, "mosaic_teacher_commitment_delta")
        pc = _maybe_get(outputs, "mosaic_commitment_logits")
        if (
            self.aux_commitment_weight > 0.0
            and isinstance(tc, Tensor)
            and isinstance(pc, Tensor)
            and pc.ndim == 3
            and int(pc.size(-1)) == 3
            and tc.shape == pc.shape[:2]
        ):
            labels_raw = tc.long()
            mask = labels_raw != -100
            if bool(mask.any().item()):
                v = labels_raw[mask]
                ok = (v == -1) | (v == 0) | (v == 1)
                if not bool(ok.all().item()):
                    raise ValueError(
                        "mosaic_teacher_commitment_delta must be in {-1,0,1} or -100 to ignore"
                    )
                labels = labels_raw.clone()
                labels[mask] = labels_raw[mask] + 1  # {-1,0,1}->{0,1,2}
                ce = F.cross_entropy(
                    pc.float().view(-1, 3),
                    labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss + float(self.aux_commitment_weight) * ce

        # Register supervision (optional).
        trg = _maybe_get(batch, "mosaic_teacher_reg_write_gate")
        prg = _maybe_get(outputs, "mosaic_reg_write_gate_logits")
        if (
            self.aux_reg_gate_weight > 0.0
            and isinstance(trg, Tensor)
            and isinstance(prg, Tensor)
            and trg.shape == prg.shape
        ):
            mask = trg >= 0
            if bool(mask.any().item()):
                rg = F.binary_cross_entropy_with_logits(prg[mask].float(), trg[mask].float())
                loss = loss + float(self.aux_reg_gate_weight) * rg

        trs = _maybe_get(batch, "mosaic_teacher_reg_sel")
        prs = _maybe_get(outputs, "mosaic_reg_sel_logits")
        if (
            self.aux_reg_sel_weight > 0.0
            and isinstance(trs, Tensor)
            and isinstance(prs, Tensor)
            and prs.ndim == 3
            and trs.shape == prs.shape[:2]
        ):
            rs = F.cross_entropy(
                prs.float().view(-1, int(prs.size(-1))),
                trs.long().view(-1),
                ignore_index=-1,
            )
            loss = loss + float(self.aux_reg_sel_weight) * rs

        # Contrastive auxiliary from the model (optional scalar).
        cl = _maybe_get(outputs, "mosaic_contrastive_loss")
        if self.aux_contrastive_weight > 0.0 and isinstance(cl, Tensor):
            # Expect scalar tensor; if not, reduce.
            cval = cl.float().mean()
            loss = loss + float(self.aux_contrastive_weight) * cval

        # State-decay regularizer (scalar), accumulated across stacked MOSAIC blocks.
        if self.aux_state_decay_weight > 0.0:
            sdl = _maybe_get(outputs, "mosaic_state_decay_reg_loss")
            if not isinstance(sdl, Tensor):
                raise KeyError(
                    "aux_state_decay_weight > 0 requires outputs['mosaic_state_decay_reg_loss']"
                )
            if int(sdl.numel()) != 1:
                raise ValueError(
                    f"mosaic_state_decay_reg_loss must be a scalar Tensor, got shape {tuple(sdl.shape)}"
                )
            loss = loss + float(self.aux_state_decay_weight) * sdl.float().reshape(())

        return loss

    def metrics(self, *, outputs: TensorDict, batch: TensorDict, loss: Tensor) -> MetricDict:
        """Expose per-component losses for debugging new memory behaviors."""
        m: MetricDict = {
            "loss_total": float(loss.detach()),
        }
        try:
            base = super().loss(outputs=outputs, batch=batch)
            m["lm_ce"] = float(base.detach())
        except Exception as e:
            raise RuntimeError(f"Failed to compute LM CE loss: {e}") from e

        tg = _maybe_get(batch, "mosaic_teacher_write_gate")
        pg = _maybe_get(outputs, "mosaic_write_gate_logits")
        tr = _maybe_get(batch, "mosaic_teacher_read_bucket")
        tw = _maybe_get(batch, "mosaic_teacher_write_bucket")
        pr = _maybe_get(outputs, "mosaic_read_bit_logits")
        pw = _maybe_get(outputs, "mosaic_write_bit_logits")
        vqr = _maybe_get(outputs, "mosaic_vq_read_logits")
        vqw = _maybe_get(outputs, "mosaic_vq_write_logits")
        tu = _maybe_get(batch, "mosaic_teacher_write_utility")
        pu = _maybe_get(outputs, "mosaic_write_utility_logits")
        to = _maybe_get(batch, "mosaic_teacher_opcode")
        po = _maybe_get(outputs, "mosaic_opcode_logits")
        tc = _maybe_get(batch, "mosaic_teacher_commitment_delta")
        pc = _maybe_get(outputs, "mosaic_commitment_logits")
        trg = _maybe_get(batch, "mosaic_teacher_reg_write_gate")
        prg = _maybe_get(outputs, "mosaic_reg_write_gate_logits")
        trs = _maybe_get(batch, "mosaic_teacher_reg_sel")
        prs = _maybe_get(outputs, "mosaic_reg_sel_logits")
        cl = _maybe_get(outputs, "mosaic_contrastive_loss")
        sdl = _maybe_get(outputs, "mosaic_state_decay_reg_loss")

        # Gate imitation (BCE).
        gate_loss = None
        if isinstance(tg, Tensor) and isinstance(pg, Tensor) and tg.shape == pg.shape:
            mask = tg >= 0
            if bool(mask.any().item()):
                try:
                    gate_loss = F.binary_cross_entropy_with_logits(pg[mask].float(), tg[mask].float())
                except Exception as e:
                    raise RuntimeError(f"Failed to compute gate loss: {e}") from e

        if gate_loss is not None:
            m["aux_gate_bce"] = float(gate_loss.detach())
            m["aux_gate_weighted"] = float((gate_loss * float(self.aux_gate_weight)).detach())

        bits_r = None
        bits_w = None
        if isinstance(tr, Tensor) and isinstance(pr, Tensor):
            try:
                bits_r = self._bits_loss(tr, pr)
            except Exception as e:
                raise RuntimeError(f"Failed to compute bits loss: {e}") from e

        if isinstance(tw, Tensor) and isinstance(pw, Tensor):
            try:
                bits_w = self._bits_loss(tw, pw)
            except Exception as e:
                raise RuntimeError(f"Failed to compute bits loss: {e}") from e

        if bits_r is not None:
            m["aux_bits_bce_read"] = float(bits_r.detach())
        if bits_w is not None:
            m["aux_bits_bce_write"] = float(bits_w.detach())
        if bits_r is not None or bits_w is not None:
            val = (bits_r if bits_r is not None else 0.0) + (bits_w if bits_w is not None else 0.0)
            if isinstance(val, Tensor):
                m["aux_bits_weighted"] = float((val * float(self.aux_bits_weight)).detach())

        vq_r = None
        vq_w = None
        if isinstance(tr, Tensor) and isinstance(vqr, Tensor):
            try:
                vq_r = self._vq_ce(tr, vqr)
            except Exception as e:
                raise RuntimeError(f"Failed to compute VQ CE loss: {e}") from e

        if isinstance(tw, Tensor) and isinstance(vqw, Tensor):
            try:
                vq_w = self._vq_ce(tw, vqw)
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
                except Exception as e:
                    raise RuntimeError(f"Failed to compute utility loss: {e}") from e

        if util_loss is not None:
            m["aux_utility_bce"] = float(util_loss.detach())
            m["aux_utility_weighted"] = float((util_loss * float(self.aux_utility_weight)).detach())

        if (
            isinstance(to, Tensor)
            and isinstance(po, Tensor)
            and po.ndim == 3
            and to.shape == po.shape[:2]
        ):
            try:
                op_ce = F.cross_entropy(
                    po.float().view(-1, int(po.size(-1))),
                    to.long().view(-1),
                    ignore_index=-1,
                )
                m["aux_opcode_ce"] = float(op_ce.detach())
                m["aux_opcode_weighted"] = float((op_ce * float(self.aux_opcode_weight)).detach())
            except Exception as e:
                raise RuntimeError(f"Failed to compute opcode CE loss: {e}") from e

        if (
            isinstance(tc, Tensor)
            and isinstance(pc, Tensor)
            and pc.ndim == 3
            and int(pc.size(-1)) == 3
            and tc.shape == pc.shape[:2]
        ):
            labels_raw = tc.long()
            mask = labels_raw != -100
            if bool(mask.any().item()):
                v = labels_raw[mask]
                ok = (v == -1) | (v == 0) | (v == 1)
                if not bool(ok.all().item()):
                    raise ValueError(
                        "mosaic_teacher_commitment_delta must be in {-1,0,1} or -100 to ignore"
                    )
                labels = labels_raw.clone()
                labels[mask] = labels_raw[mask] + 1
                try:
                    c_ce = F.cross_entropy(
                        pc.float().view(-1, 3),
                        labels.view(-1),
                        ignore_index=-100,
                    )
                    m["aux_commitment_ce"] = float(c_ce.detach())
                    m["aux_commitment_weighted"] = float((c_ce * float(self.aux_commitment_weight)).detach())
                except Exception as e:
                    raise RuntimeError(f"Failed to compute commitment CE loss: {e}") from e

        if isinstance(trg, Tensor) and isinstance(prg, Tensor) and trg.shape == prg.shape:
            mask = trg >= 0
            if bool(mask.any().item()):
                try:
                    rg = F.binary_cross_entropy_with_logits(prg[mask].float(), trg[mask].float())
                    m["aux_reg_gate_bce"] = float(rg.detach())
                    m["aux_reg_gate_weighted"] = float((rg * float(self.aux_reg_gate_weight)).detach())
                except Exception as e:
                    raise RuntimeError(f"Failed to compute reg gate loss: {e}") from e

        if (
            isinstance(trs, Tensor)
            and isinstance(prs, Tensor)
            and prs.ndim == 3
            and trs.shape == prs.shape[:2]
        ):
            try:
                rs = F.cross_entropy(
                    prs.float().view(-1, int(prs.size(-1))),
                    trs.long().view(-1),
                    ignore_index=-1,
                )
                m["aux_reg_sel_ce"] = float(rs.detach())
                m["aux_reg_sel_weighted"] = float((rs * float(self.aux_reg_sel_weight)).detach())
            except Exception as e:
                raise RuntimeError(f"Failed to compute reg sel loss: {e}") from e

        if isinstance(cl, Tensor):
            try:
                cval = cl.float().mean()
                m["aux_contrastive"] = float(cval.detach())
                m["aux_contrastive_weighted"] = float((cval * float(self.aux_contrastive_weight)).detach())
            except Exception as e:
                raise RuntimeError(f"Failed to compute contrastive loss: {e}") from e

        if isinstance(sdl, Tensor):
            if int(sdl.numel()) != 1:
                raise ValueError(
                    f"mosaic_state_decay_reg_loss must be a scalar Tensor, got shape {tuple(sdl.shape)}"
                )
            m["aux_state_decay_reg"] = float(sdl.detach().float().reshape(()))
            m["aux_state_decay_reg_weighted"] = float(
                (sdl.detach().float().reshape(()) * float(self.aux_state_decay_weight))
            )

        return m


class MosaicEventPrediction(MosaicNextTokenWithAuxObjective):
    """Event-centric alias for MOSAIC control-surface supervision.

    This objective keeps next-token cross-entropy as the internal "VM step"
    training signal, while adding auxiliary losses that supervise the model's
    control surface (memory gates, routing logits, opcode logits, register gates).
    """
