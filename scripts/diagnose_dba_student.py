#!/usr/bin/env python3
"""
Diagnose a DBA upcycled student quickly (no full benchmarks).

This script is meant to answer, in minutes:
- Are the student's logits numerically sane?
- Are DBA gates saturated (all semantic or all geometric)?
- Which layers differ most between teacher and student outputs?

Typical usage (from repo root):
  python3.12 scripts/diagnose_dba_student.py \\
    --teacher-ckpt hf://meta-llama/Llama-3.2-1B \\
    --student-ckpt runs/paper/finetune_global_final.pt \\
    --dataset artifacts/datasets/fineweb_llama/fineweb_llama_1b.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

# Allow running as a script from repo root (no installed package required).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.state_dict import AdapterStateDictTransformer  # noqa: E402
from benchmark.utils import get_model_vocab_size  # noqa: E402
from carmath import weight_dtype  # noqa: E402
from config.model import ModelConfig  # noqa: E402
from console import logger  # noqa: E402
from data.npy import NpyDataset  # noqa: E402
from loader.checkpoint import CheckpointBuilder  # noqa: E402
from loader.checkpoint.error import CheckpointError  # noqa: E402
from loader.hf import HFLoader  # noqa: E402
from model import Model  # noqa: E402
from model.trace import Trace, TraceStop  # noqa: E402
from trainer.initializers.default import _make_teacher_model_config  # noqa: E402


def _infer_n_heads(d_model: int) -> int:
    # Heuristic: prefer larger head counts up to 32.
    for h in (32, 24, 16, 12, 8, 6, 4, 3, 2, 1):
        if d_model % h == 0:
            return int(h)
    return 1


def _find_embedding_weight(state: dict[str, Tensor]) -> tuple[str, Tensor]:
    """Best-effort locate token embedding weight in a state_dict.

    Caramba checkpoints may vary in key naming across experiments (especially older runs).
    We use a small list of common names, then fall back to a heuristic scan.
    """
    # Common exact keys in this repo / HF-ish conventions.
    candidates = [
        "embedder.token_embedding.weight",
        "embedder.token_embeddings.weight",
        "embedder.embedding.weight",
        "embedder.embeddings.weight",
        "embed_tokens.weight",
        "tok_embeddings.weight",
        "model.embed_tokens.weight",
        "transformer.wte.weight",
    ]

    # Allow prefixes (e.g., "model." or "module.").
    for k in candidates:
        if k in state and isinstance(state[k], Tensor):
            return k, state[k]
        for kk, vv in state.items():
            if kk.endswith(k) and isinstance(vv, Tensor):
                return kk, vv

    # Heuristic: find a 2D weight that looks like [vocab, d_model] with a large-ish vocab.
    best_key = None
    best_tensor: Tensor | None = None
    best_vocab = -1
    for k, v in state.items():
        if not isinstance(v, Tensor) or v.ndim != 2:
            continue
        if not (k.endswith(".weight") or "embedding" in k or "embed" in k):
            continue
        vocab, d_model = int(v.shape[0]), int(v.shape[1])
        # Prefer plausible vocabs.
        if vocab < 1024 or d_model < 64:
            continue
        if vocab > best_vocab:
            best_vocab = vocab
            best_key = k
            best_tensor = v

    if best_key is None or best_tensor is None:
        raise ValueError(
            "Cannot infer model config: missing token embedding weight in state_dict "
            "(tried common keys and heuristic scan)."
        )
    return best_key, best_tensor


def _infer_transformer_payload_from_state(
    *,
    state: dict[str, Tensor],
    mode: str,
) -> dict[str, Any]:
    """Infer a minimal TransformerModel payload from a Caramba system_state_dict.

    This is meant for diagnostics only (to load weights + run a single batch),
    not for reproducing the exact training config.

    Args:
        state: Model/system state dict (Caramba keyspace, e.g. embedder.token_embedding.weight)
        mode: "standard" or "decoupled"
    """
    emb_key, emb = _find_embedding_weight(state)
    if emb.ndim != 2:
        raise ValueError(f"Unexpected embedding weight shape for {emb_key}: {tuple(emb.shape)}")
    vocab_size, d_model = int(emb.shape[0]), int(emb.shape[1])

    # Infer number of attention blocks (one qkv_proj per block in standard attention).
    qkv_keys = [k for k in state.keys() if k.endswith(".qkv_proj.weight")]
    n_layers = int(len(qkv_keys))
    if n_layers <= 0:
        # Decoupled attention uses q_sem/q_geo etc; fall back to counting out_proj occurrences.
        out_keys = [k for k in state.keys() if k.endswith(".out_proj.weight")]
        n_layers = int(len(out_keys))

    # Infer SwiGLU d_ff from w_gate_up weight if present (shape: [2*d_ff, d_model]).
    d_ff = None
    for k, w in state.items():
        if k.endswith(".w_gate_up.weight") and isinstance(w, Tensor) and w.ndim == 2:
            if int(w.shape[1]) == d_model and int(w.shape[0]) % 2 == 0:
                d_ff = int(w.shape[0] // 2)
                break
    if d_ff is None:
        # Conservative fallback
        d_ff = int(4 * d_model)

    n_heads = _infer_n_heads(d_model)

    attn_extra: dict[str, Any] = {
        "d_model": d_model,
        "n_heads": n_heads,
        "rope_enabled": True,
        "rope_base": 10000.0,
        "is_causal": True,
        "dropout_p": 0.0,
    }

    mode_s = str(mode).lower().strip()
    if mode_s == "decoupled":
        # Infer sem/geo/attn dims from projection shapes when possible.
        sem_dim = geo_dim = attn_dim = None
        for k, w in state.items():
            if k.endswith(".q_sem.weight") and w.ndim == 2 and int(w.shape[1]) == d_model:
                sem_dim = int(w.shape[0])
                break
        for k, w in state.items():
            if k.endswith(".q_geo.weight") and w.ndim == 2 and int(w.shape[1]) == d_model:
                geo_dim = int(w.shape[0])
                break
        for k, w in state.items():
            if k.endswith(".v_proj.weight") and w.ndim == 2 and int(w.shape[1]) == d_model:
                attn_dim = int(w.shape[0])
                break
        # Fallbacks if we couldn't find decoupled projections for some reason.
        sem_dim = int(sem_dim or (d_model // 8))
        geo_dim = int(geo_dim or (d_model // 2))
        attn_dim = int(attn_dim or (sem_dim + geo_dim))

        attn_extra.update(
            {
                "mode": "decoupled",
                "attn_dim": attn_dim,
                "sem_dim": sem_dim,
                "geo_dim": geo_dim,
                # Gate param exists only when enabled; infer from presence in checkpoint.
                "decoupled_gate": any("decoupled_gate_logit" in k for k in state.keys()),
            }
        )
    else:
        attn_extra.update({"mode": "standard"})

    return {
        "type": "TransformerModel",
        "tied_embeddings": False,
        "embedder": {"type": "token", "vocab_size": vocab_size, "d_model": d_model},
        "topology": {
            "type": "StackedTopology",
            "layers": [
                {
                    "type": "NestedTopology",
                    "repeat": n_layers,
                    "layers": [
                        {
                            "type": "ResidualTopology",
                            "layers": [
                                {"type": "RMSNormLayer", "d_model": d_model, "eps": 1e-5},
                                {"type": "AttentionLayer", **attn_extra},
                            ],
                        },
                        {
                            "type": "ResidualTopology",
                            "layers": [
                                {"type": "RMSNormLayer", "d_model": d_model, "eps": 1e-5},
                                {"type": "SwiGLULayer", "d_model": d_model, "d_ff": d_ff, "bias": False},
                            ],
                        },
                    ],
                },
                {"type": "RMSNormLayer", "d_model": d_model, "eps": 1e-5},
                {"type": "LinearLayer", "d_in": d_model, "d_out": vocab_size, "bias": False},
            ],
        },
    }

def _fmt_exc(e: BaseException) -> str:
    msg = str(e).strip()
    if msg:
        return f"{type(e).__name__}: {msg}"
    return type(e).__name__


def _extract_state_dict_from_obj(obj: object) -> tuple[dict[str, Tensor], str]:
    """Extract a tensor state_dict from common Caramba checkpoint containers."""
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint must be a dict-like object (got {type(obj).__name__})")

    # Common Caramba training checkpoint containers.
    for key in ("system_state_dict", "model_state_dict", "student_state_dict", "state_dict"):
        if key in obj and isinstance(obj[key], dict):
            sd = cast(dict[str, Tensor], obj[key])
            # Best-effort sanity check: keys must be strings; values should be tensors.
            if all(isinstance(k, str) for k in sd.keys()):
                return sd, f"container:{key}"

    # Raw tensor-only dict (CheckpointBuilder format).
    if all(isinstance(k, str) for k in obj.keys()) and all(isinstance(v, Tensor) for v in obj.values()):
        return cast(dict[str, Tensor], obj), "raw:tensor_dict"

    # Last resort: if it's a dict of string->(something), allow it (caller will likely fail later),
    # but return an informative source tag.
    if all(isinstance(k, str) for k in obj.keys()):
        return cast(dict[str, Tensor], obj), "raw:mixed_dict"

    raise TypeError("Checkpoint dict did not contain a usable state_dict key")

def _strip_prefix_if_present(sd: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


def _normalize_state_dict_keys(sd: dict[str, Tensor]) -> tuple[dict[str, Tensor], str | None]:
    """Normalize common training-time prefixes in state_dict keys.

    Examples:
    - torch.compile often wraps modules under `_orig_mod.` and checkpoints then
      store keys like `_orig_mod.embedder.token_embedding.weight`.
    - DDP can introduce `module.`.

    We strip a prefix if it dominates the keyspace (>= 90% of keys).
    """
    if not sd:
        return sd, None

    prefixes = ("_orig_mod.", "module.", "model.", "net.")
    keys = list(sd.keys())
    n = len(keys)
    for pref in prefixes:
        hits = sum(1 for k in keys if k.startswith(pref))
        if hits / float(n) >= 0.9:
            stripped = _strip_prefix_if_present(sd, pref)
            # Only accept if we still have a meaningful dict.
            if stripped and len(stripped) >= int(0.8 * n):
                return stripped, pref
    return sd, None


def _fuse_qkv_if_needed(sd: dict[str, Tensor]) -> tuple[dict[str, Tensor], int]:
    """Fuse split Q/K/V projections into packed qkv_proj if the checkpoint uses split keys.

    Some older runs / alternative attention implementations store:
      *.q_proj.weight, *.k_proj.weight, *.v_proj.weight
    while current `StandardAttentionLayer` expects:
      *.qkv_proj.weight  (packed [q;k;v] along dim=0)
    """
    if not sd:
        return sd, 0

    # Collect base prefixes that have q/k/v weights.
    q_keys = [k for k in sd.keys() if k.endswith(".q_proj.weight")]
    if not q_keys:
        return sd, 0

    out = dict(sd)
    fused = 0
    for qk in q_keys:
        base = qk[: -len(".q_proj.weight")]
        kk = base + ".k_proj.weight"
        vk = base + ".v_proj.weight"
        qkvk = base + ".qkv_proj.weight"
        if kk not in out or vk not in out:
            continue
        if qkvk in out:
            # Already packed; skip.
            continue

        q = out.get(qk)
        k = out.get(kk)
        v = out.get(vk)
        if not (isinstance(q, Tensor) and isinstance(k, Tensor) and isinstance(v, Tensor)):
            continue
        if not (q.ndim == 2 and k.ndim == 2 and v.ndim == 2):
            continue
        if not (q.shape[1] == k.shape[1] == v.shape[1]):
            continue

        out[qkvk] = torch.cat([q, k, v], dim=0)
        # Remove split keys so they don't show up as "unexpected".
        out.pop(qk, None)
        out.pop(kk, None)
        out.pop(vk, None)

        # Optional bias packing if present.
        qb = base + ".q_proj.bias"
        kb = base + ".k_proj.bias"
        vb = base + ".v_proj.bias"
        qkvb = base + ".qkv_proj.bias"
        if qb in out and kb in out and vb in out and qkvb not in out:
            bq = out.get(qb)
            bk = out.get(kb)
            bv = out.get(vb)
            if isinstance(bq, Tensor) and isinstance(bk, Tensor) and isinstance(bv, Tensor):
                if bq.ndim == bk.ndim == bv.ndim == 1:
                    out[qkvb] = torch.cat([bq, bk, bv], dim=0)
                    out.pop(qb, None)
                    out.pop(kb, None)
                    out.pop(vb, None)

        fused += 1

    return out, fused


def _log_load_result(label: str, model: nn.Module, res: object) -> None:
    missing = list(getattr(res, "missing_keys", []) or [])
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    total = len(model.state_dict())
    loaded = max(0, total - len(missing))
    ratio = (loaded / float(total)) if total else 0.0

    if missing:
        logger.warning(f"{label}: missing {len(missing)} keys (showing up to 8): {missing[:8]}")
    if unexpected:
        logger.warning(f"{label}: unexpected {len(unexpected)} keys (showing up to 8): {unexpected[:8]}")
    if total:
        logger.info(f"{label}: loaded_keys≈{loaded}/{total} ({ratio:.3f})")
        if ratio < 0.2:
            logger.warning(
                f"{label}: VERY low load ratio ({ratio:.3f}). "
                "This usually means the checkpoint keys are namespaced (e.g. `_orig_mod.`/`module.`) "
                "or you're loading the wrong architecture."
            )


def _state_dict_quick_summary(sd: dict[str, Tensor]) -> list[str]:
    lines: list[str] = []
    n = len(sd)
    if n == 0:
        return ["keys=0"]

    # dtype + size summaries (cheap).
    dtype_counts: dict[str, int] = {}
    total_numel = 0
    total_bytes = 0
    for v in sd.values():
        if not isinstance(v, Tensor):
            continue
        dtype_counts[str(v.dtype)] = dtype_counts.get(str(v.dtype), 0) + 1
        total_numel += int(v.numel())
        total_bytes += int(v.numel() * v.element_size())

    lines.append(f"keys={n} tensor_keys={sum(dtype_counts.values())}")
    if dtype_counts:
        dts = ", ".join(f"{k}:{v}" for k, v in sorted(dtype_counts.items(), key=lambda kv: kv[0]))
        lines.append(f"dtypes=({dts})")
    if total_numel:
        lines.append(f"params≈{total_numel:,} elems bytes≈{total_bytes/1e9:.3f}GB")
    return lines


def _key_pattern_counts(sd: dict[str, Tensor]) -> dict[str, int]:
    pats = {
        "qkv_proj": ".qkv_proj.",
        "q_sem": ".q_sem.",
        "k_sem": ".k_sem.",
        "q_geo": ".q_geo.",
        "k_geo": ".k_geo.",
        "v_proj": ".v_proj.",
        "out_proj": ".out_proj.",
        "gate_logit": "decoupled_gate_logit",
        "w_gate_up": ".w_gate_up.",
        "w_down": ".w_down.",
        "token_embedding": "token_embedding",
        "lm_head": ".weight",
    }
    out: dict[str, int] = {k: 0 for k in pats.keys()}
    for k in sd.keys():
        for name, pat in pats.items():
            if pat in k:
                out[name] += 1
    return out


def _student_config_signature(cfg: ModelConfig) -> dict[str, Any]:
    """Extract a compact, comparable signature from a ModelConfig."""
    def _to_int(x: object) -> int | None:
        if x is None:
            return None
        try:
            return int(x)  # type: ignore[arg-type]
        except Exception:
            return None

    d = cfg.model_dump()
    embed = cast(dict[str, Any], d.get("embedder", {}) or {})
    topo = cast(dict[str, Any], d.get("topology", {}) or {})
    layers = cast(list[Any], topo.get("layers", []) or [])
    n_layers = None
    # Our inferred payload uses StackedTopology -> [NestedTopology(repeat=n_layers), ...]
    for node in layers:
        if isinstance(node, dict) and node.get("type") == "NestedTopology" and "repeat" in node:
            try:
                n_layers = int(node["repeat"])
            except Exception:
                n_layers = None
            break

    # Find first attention layer payload to read dims/mode.
    attn: dict[str, Any] | None = None
    for node in layers:
        if not isinstance(node, dict):
            continue
        if node.get("type") != "NestedTopology":
            continue
        inner = node.get("layers", [])
        if not isinstance(inner, list):
            continue
        for maybe_res in inner:
            if not isinstance(maybe_res, dict):
                continue
            res_layers = maybe_res.get("layers", [])
            if not isinstance(res_layers, list):
                continue
            for leaf in res_layers:
                if isinstance(leaf, dict) and leaf.get("type") == "AttentionLayer":
                    attn = leaf
                    break
            if attn is not None:
                break
        if attn is not None:
            break

    return {
        "vocab_size": _to_int(embed.get("vocab_size", None)),
        "d_model": _to_int(embed.get("d_model", None)),
        "n_layers": n_layers,
        "attn_mode": (attn or {}).get("mode", None),
        "sem_dim": (attn or {}).get("sem_dim", None),
        "geo_dim": (attn or {}).get("geo_dim", None),
        "attn_dim": (attn or {}).get("attn_dim", None),
        "decoupled_gate": (attn or {}).get("decoupled_gate", None),
    }


def _load_checkpoint_state(
    ckpt: str | Path,
    *,
    label: str,
    unsafe_pickle_load: bool,
    prefer_checkpoint_builder: bool,
) -> dict[str, Tensor]:
    p = Path(ckpt) if not isinstance(ckpt, Path) else ckpt

    # 1) Prefer safe torch.load for local pytorch training checkpoints (.pt/.pth/.bin),
    # because CheckpointBuilder expects a tensor-only dict and will emit "checkpoint error"
    # for full training checkpoints (optimizer/state/metadata).
    if not prefer_checkpoint_builder and p.exists() and p.is_file():
        try:
            obj = torch.load(p, map_location="cpu", weights_only=True)
        except Exception as e:
            if not unsafe_pickle_load:
                raise
            logger.warning(f"{label}: torch.load(weights_only=True) failed; retrying unsafe pickle load: {_fmt_exc(e)}")
            obj = torch.load(p, map_location="cpu", weights_only=False)

        sd, src = _extract_state_dict_from_obj(obj)
        sd2, stripped = _normalize_state_dict_keys(sd)
        if stripped is not None:
            logger.info(f"{label}: normalized state_dict keys by stripping prefix {stripped!r}")
        sd = sd2
        logger.success(f"{label}: loaded via torch.load ({src}); " + " ".join(_state_dict_quick_summary(sd)))
        return sd

    # 2) Generic format handling (directories, safetensors, HF snapshots, tensor-only files).
    try:
        sd = cast(dict[str, Tensor], CheckpointBuilder().load(p))
        logger.success(f"{label}: loaded via CheckpointBuilder (tensor dict); " + " ".join(_state_dict_quick_summary(sd)))
        return sd
    except CheckpointError as e:
        # This will already have logged "checkpoint error" due to CoreError, so add context here.
        logger.warning(
            f"{label}: CheckpointBuilder could not load as tensor-only state_dict ({_fmt_exc(e)}). "
            "This often means the file is a full training checkpoint (optimizer/metadata) rather than a raw state_dict."
        )
    except Exception as e:
        logger.warning(f"{label}: CheckpointBuilder load failed ({_fmt_exc(e)}); falling back to torch.load()")

    # 3) Fallback: torch.load and extract nested state dict.
    try:
        obj = torch.load(p, map_location="cpu", weights_only=True)
    except Exception as e:
        if not unsafe_pickle_load:
            raise
        logger.warning(f"{label}: torch.load(weights_only=True) failed; retrying unsafe pickle load: {_fmt_exc(e)}")
        obj = torch.load(p, map_location="cpu", weights_only=False)

    sd, src = _extract_state_dict_from_obj(obj)
    sd2, stripped = _normalize_state_dict_keys(sd)
    if stripped is not None:
        logger.info(f"{label}: normalized state_dict keys by stripping prefix {stripped!r}")
    sd = sd2
    logger.success(f"{label}: loaded via torch.load fallback ({src}); " + " ".join(_state_dict_quick_summary(sd)))
    return sd


def _load_teacher_state(ckpt: str, *, unsafe_pickle_load: bool = False) -> tuple[dict[str, Tensor], bool]:
    """Load teacher weights and report whether Llama adapter is required."""
    if ckpt.startswith("hf://"):
        p = HFLoader(repo_id=ckpt[5:]).load()
        # HF snapshots are directories; prefer the builder.
        sd = _load_checkpoint_state(
            p,
            label="teacher_ckpt",
            unsafe_pickle_load=unsafe_pickle_load,
            prefer_checkpoint_builder=True,
        )
        return sd, True

    # Local runs: prefer torch.load first to avoid confusing "checkpoint error" spam for full training checkpoints.
    sd = _load_checkpoint_state(
        ckpt,
        label="teacher_ckpt",
        unsafe_pickle_load=unsafe_pickle_load,
        prefer_checkpoint_builder=False,
    )
    return sd, False


def _load_student_state(ckpt_path: Path, *, unsafe_pickle_load: bool) -> dict[str, Tensor]:
    # Student is typically a local .pt; prefer torch.load (see _load_checkpoint_state rationale).
    return _load_checkpoint_state(
        ckpt_path,
        label="student_ckpt",
        unsafe_pickle_load=unsafe_pickle_load,
        prefer_checkpoint_builder=False,
    )


def _logit_tensor(out: object) -> Tensor:
    if isinstance(out, tuple) and out:
        out = out[0]
    if hasattr(out, "logits"):
        out = getattr(out, "logits")
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Model output is not a Tensor (got {type(out).__name__})")
    return out


def _stats(name: str, t: Tensor) -> str:
    tf = t.detach().float()
    finite = torch.isfinite(tf)
    n = int(tf.numel())
    nf = int((~finite).sum().item())
    if n == 0:
        return f"{name}: empty"
    if nf == n:
        return f"{name}: all_nonfinite n={n} dtype={t.dtype} device={t.device}"
    x = tf[finite]
    return (
        f"{name}: dtype={t.dtype} device={t.device} shape={tuple(t.shape)} "
        f"nonfinite={nf}/{n} min={float(x.min().item()):.5g} max={float(x.max().item()):.5g} "
        f"mean={float(x.mean().item()):.5g} std={float(x.std(unbiased=False).item()):.5g} "
        f"max_abs={float(x.abs().max().item()):.5g}"
    )


def _entropy_lastpos(logits: Tensor) -> float:
    # logits: (B,T,V) -> entropy at last position
    last = logits[:, -1, :].float()
    probs = torch.softmax(last, dim=-1)
    eps = 1e-9
    ent = -(probs * (probs + eps).log()).sum(dim=-1).mean()
    return float(ent.item())


def _logits_agreement_report(teacher_logits: Tensor, student_logits: Tensor, *, topk: int = 10) -> list[str]:
    """Cheap agreement metrics (mostly last-position to keep it fast)."""
    lines: list[str] = []
    if teacher_logits.shape != student_logits.shape:
        return [f"logits shape mismatch teacher={tuple(teacher_logits.shape)} student={tuple(student_logits.shape)}"]

    lt = teacher_logits.detach().float()
    ls = student_logits.detach().float()
    b, t, v = lt.shape
    _ = b, v  # silence unused in case we trim later

    # Top-1 agreement across the whole sequence (cheap: argmax only).
    at = lt.argmax(dim=-1)
    as_ = ls.argmax(dim=-1)
    agree = float((at == as_).float().mean().item())
    lines.append(f"argmax_agreement(all_positions)={agree:.4f}")

    # Last-position KL + top-k overlap.
    lt_last = lt[:, -1, :]
    ls_last = ls[:, -1, :]
    logpt = torch.log_softmax(lt_last, dim=-1)
    logps = torch.log_softmax(ls_last, dim=-1)
    pt = torch.softmax(lt_last, dim=-1)
    kl_ts = float((pt * (logpt - logps)).sum(dim=-1).mean().item())
    lines.append(f"kl(teacher||student,last_pos)={kl_ts:.6f}")

    k = int(max(1, topk))
    top_t = torch.topk(lt_last, k=k, dim=-1).indices
    top_s = torch.topk(ls_last, k=k, dim=-1).indices
    overlap = []
    for i in range(top_t.shape[0]):
        overlap.append(len(set(top_t[i].tolist()).intersection(set(top_s[i].tolist()))) / float(k))
    lines.append(f"top{k}_overlap(last_pos)={sum(overlap)/len(overlap):.4f}")

    return lines


def _ce_window_report(logits: Tensor, targets: Tensor, *, window: int) -> float:
    import torch.nn.functional as F

    w = int(max(1, window))
    l = logits[:, -w:, :].float()
    y = targets[:, -w:].long()
    return float(F.cross_entropy(l.reshape(-1, l.size(-1)), y.reshape(-1)).detach().item())

def _pairwise_lastpos_report(a_logits: Tensor, b_logits: Tensor, *, label: str, topk: int = 10) -> list[str]:
    """Agreement metrics for two logits tensors (mostly last-pos to keep it fast)."""
    if a_logits.shape != b_logits.shape:
        return [f"{label}: logits shape mismatch a={tuple(a_logits.shape)} b={tuple(b_logits.shape)}"]

    a = a_logits.detach().float()
    b = b_logits.detach().float()

    # Top-1 agreement across full sequence.
    aa = a.argmax(dim=-1)
    bb = b.argmax(dim=-1)
    agree = float((aa == bb).float().mean().item())

    # Last-position KL(a||b).
    a_last = a[:, -1, :]
    b_last = b[:, -1, :]
    logpa = torch.log_softmax(a_last, dim=-1)
    logpb = torch.log_softmax(b_last, dim=-1)
    pa = torch.softmax(a_last, dim=-1)
    kl_ab = float((pa * (logpa - logpb)).sum(dim=-1).mean().item())

    # Last-position top-k overlap.
    k = int(max(1, topk))
    top_a = torch.topk(a_last, k=k, dim=-1).indices
    top_b = torch.topk(b_last, k=k, dim=-1).indices
    overlap = []
    for i in range(top_a.shape[0]):
        overlap.append(len(set(top_a[i].tolist()).intersection(set(top_b[i].tolist()))) / float(k))
    return [
        f"{label}: argmax_agreement(all_positions)={agree:.4f}",
        f"{label}: kl(a||b,last_pos)={kl_ab:.6f}",
        f"{label}: top{k}_overlap(last_pos)={sum(overlap)/len(overlap):.4f}",
    ]


def _maybe_dashboard(
    *,
    dataset_name: str,
    dataset_path: str,
    pre: dict[str, Any] | None,
    post: dict[str, Any],
    ft_delta: dict[str, Any] | None,
    plot: bool,
    plot_out: str | None,
    plot_dpi: int,
) -> None:
    if not plot and not plot_out:
        return

    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"matplotlib unavailable; skipping dashboard: {_fmt_exc(e)}")
        return

    # Use a non-interactive backend when saving to file (common for remote runs).
    if plot_out:
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass

    def layer_rms_map(res: dict[str, Any]) -> dict[int, float]:
        m: dict[int, float] = {}
        for i, _l1, l2 in cast(list[tuple[int, float, float]], res.get("layer_scores", [])):
            m[int(i)] = float(l2)
        return m

    post_map = layer_rms_map(post)
    pre_map = layer_rms_map(pre) if pre else {}

    # Choose a layer set to visualize (top by post RMS).
    top_layers = sorted(post_map.items(), key=lambda kv: kv[1], reverse=True)[:12]
    layer_ids = [i for i, _ in top_layers]
    post_vals = [post_map.get(i, 0.0) for i in layer_ids]
    pre_vals = [pre_map.get(i, 0.0) for i in layer_ids]

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    x = list(range(len(layer_ids)))
    w = 0.38
    if pre:
        ax0.bar([xi - w / 2 for xi in x], pre_vals, width=w, label="pre-ft", alpha=0.85)
        ax0.bar([xi + w / 2 for xi in x], post_vals, width=w, label="post-ft", alpha=0.85)
        ax0.legend()
    else:
        ax0.bar(x, post_vals, width=0.7, label="post-ft", alpha=0.9)
        ax0.legend()
    ax0.set_title("Attention output mismatch (RMS |Δ|) — top layers by post-ft")
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"L{i:02d}" for i in layer_ids], rotation=0)
    ax0.set_ylabel("RMS |Δ|")

    ax1 = fig.add_subplot(gs[0, 1])
    # CE + entropy summary
    stages = []
    t_ce = []
    s_ce = []
    t_ent = []
    s_ent = []
    if pre:
        stages.append("pre")
        t_ce.append(float(pre.get("teacher_ce", float("nan"))))
        s_ce.append(float(pre.get("student_ce", float("nan"))))
        t_ent.append(float(pre.get("teacher_entropy", float("nan"))))
        s_ent.append(float(pre.get("student_entropy", float("nan"))))
    stages.append("post")
    t_ce.append(float(post.get("teacher_ce", float("nan"))))
    s_ce.append(float(post.get("student_ce", float("nan"))))
    t_ent.append(float(post.get("teacher_entropy", float("nan"))))
    s_ent.append(float(post.get("student_entropy", float("nan"))))

    xx = list(range(len(stages)))
    ax1.plot(xx, t_ce, marker="o", label="teacher CE(window)")
    ax1.plot(xx, s_ce, marker="o", label="student CE(window)")
    ax1.set_xticks(xx)
    ax1.set_xticklabels(stages)
    ax1.set_title("CE (window) across stages")
    ax1.set_ylabel("Cross-entropy")
    ax1.legend()
    ax1b = ax1.twinx()
    ax1b.plot(xx, t_ent, marker="x", linestyle="--", label="teacher entropy(last)")
    ax1b.plot(xx, s_ent, marker="x", linestyle="--", label="student entropy(last)")
    ax1b.set_ylabel("Entropy(last position)")

    ax2 = fig.add_subplot(gs[1, 0])
    # Teacher||Student KL + argmax agreement (separate y-scales for readability)
    labels = (["pre", "post"] if pre else ["post"])
    xx2 = list(range(len(labels)))
    kls = []
    ags = []
    for lab in labels:
        src = pre if (lab == "pre") else post
        kls.append(float(cast(dict[str, Any], src).get("ts_kl_last", float("nan"))))
        ags.append(float(cast(dict[str, Any], src).get("ts_argmax_agree", float("nan"))))

    ax2.plot(xx2, kls, marker="o", label="KL(teacher||student,last)")
    ax2.set_xticks(xx2)
    ax2.set_xticklabels(labels)
    ax2.set_title("Teacher↔Student divergence across stages")
    ax2.set_ylabel("KL(teacher||student,last)")
    ax2.legend(loc="upper left")
    ax2b = ax2.twinx()
    ax2b.plot(xx2, ags, marker="x", linestyle="--", color="tab:orange", label="argmax agreement(all)")
    ax2b.set_ylabel("argmax agreement (all positions)")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    lines = []
    if ft_delta:
        lines.append("Finetune delta (pre→post)")
        lines.append(f"Teacher: ΔCE={ft_delta.get('teacher_dce', 'n/a')}, ΔKL(pre||post,last)={ft_delta.get('teacher_kl_pre_post', 'n/a')}")
        lines.append(f"Student: ΔCE={ft_delta.get('student_dce', 'n/a')}, ΔKL(pre||post,last)={ft_delta.get('student_kl_pre_post', 'n/a')}")
        lines.append(f"ΔKL(teacher||student,last): {ft_delta.get('d_kl_ts', 'n/a')}")
        lines.append(f"Δargmax_agree(teacher,student): {ft_delta.get('d_agree_ts', 'n/a')}")
    else:
        lines.append("Finetune delta: (provide --teacher-pre-ckpt and --student-pre-ckpt)")
    ax3.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    fig.suptitle(f"DBA Diagnose Dashboard • {dataset_name}", fontsize=14)
    ax3.text(0.02, 0.04, f"dataset: {dataset_path}", va="bottom", ha="left", fontsize=9, family="monospace")

    if plot_out:
        try:
            fig.savefig(str(plot_out), dpi=int(plot_dpi))
            logger.success(f"Saved dashboard to {plot_out}")
        except Exception as e:
            logger.warning(f"Failed to save dashboard: {_fmt_exc(e)}")
    if plot and not plot_out:
        try:
            plt.show()
        except Exception as e:
            logger.warning(f"plt.show() failed (headless?): {_fmt_exc(e)}")
    plt.close(fig)


def _run_pair(
    *,
    stage: str,
    teacher_ckpt: str,
    student_ckpt: str,
    device: torch.device,
    dt: torch.dtype,
    x: Tensor,
    y: Tensor,
    topk_layers: int,
    logits_topk: int,
    loss_window: int,
    unsafe_pickle_load: bool,
) -> dict[str, Any]:
    """Run all diagnostics for one (teacher, student) checkpoint pair.

    Returns:
        A dict of metrics + logits for follow-on comparisons/plots.
    """
    logger.header(f"Stage: {stage}", "checkpoint pair")

    logger.info(f"Loading student checkpoint: {student_ckpt}")
    student_sd = _load_student_state(Path(student_ckpt), unsafe_pickle_load=bool(unsafe_pickle_load))
    pat = _key_pattern_counts(student_sd)
    logger.info(
        "student_ckpt key patterns: "
        + " ".join(f"{k}={v}" for k, v in pat.items() if k in {"qkv_proj", "q_sem", "q_geo", "out_proj", "gate_logit"})
    )

    # Infer student config from its checkpoint; teacher is rewritten to standard attention.
    student_payload = _infer_transformer_payload_from_state(state=student_sd, mode="decoupled")
    student_cfg = ModelConfig.model_validate(student_payload)
    sig = _student_config_signature(student_cfg)
    logger.info(
        "inferred_student_cfg: "
        + " ".join(f"{k}={v}" for k, v in sig.items() if v is not None)
    )
    teacher_cfg = _make_teacher_model_config(student_cfg)

    teacher = Model(teacher_cfg).to(device=device, dtype=dt).eval()
    teacher_state, needs_llama_adapter = _load_teacher_state(
        str(teacher_ckpt), unsafe_pickle_load=bool(unsafe_pickle_load)
    )
    if needs_llama_adapter:
        AdapterStateDictTransformer.llama(dba_init="svd").apply(model=teacher, state_dict=teacher_state)
    else:
        # Some checkpoints store split q/k/v weights; fuse them into qkv_proj if needed.
        teacher_state2, fused = _fuse_qkv_if_needed(teacher_state)
        if fused:
            logger.info(f"teacher_ckpt: fused split q/k/v into qkv_proj for {fused} layers")
        teacher_state = teacher_state2
        res = teacher.load_state_dict(teacher_state, strict=False)
        _log_load_result("teacher_load", teacher, res)

    student = Model(student_cfg).to(device=device, dtype=dt).eval()
    res_s = student.load_state_dict(student_sd, strict=False)
    _log_load_result("student_load", student, res_s)

    v_teacher = get_model_vocab_size(teacher, default=32000)
    v_student = get_model_vocab_size(student, default=32000)
    logger.info(f"vocab teacher={v_teacher} student={v_student}")
    mx = int(max(x.max().item(), y.max().item()))
    if mx >= v_student:
        logger.warning(f"batch max_id={mx} >= student_vocab={v_student} (tokenizer/checkpoint mismatch?)")

    with torch.no_grad():
        lt = _logit_tensor(teacher(x))
        ls = _logit_tensor(student(x))

    teacher_entropy = _entropy_lastpos(lt)
    student_entropy = _entropy_lastpos(ls)
    logger.info(_stats("teacher_logits", lt))
    logger.info(_stats("student_logits", ls))
    logger.info(f"entropy_lastpos teacher={teacher_entropy:.3f} student={student_entropy:.3f}")

    agree_lines = _logits_agreement_report(lt, ls, topk=int(logits_topk))
    for line in agree_lines:
        logger.info(line)

    # Parse out the core metrics we care about for later delta/plotting.
    ts_argmax_agree = float("nan")
    ts_kl_last = float("nan")
    ts_topk_overlap_last = float("nan")
    for s in agree_lines:
        if s.startswith("argmax_agreement"):
            ts_argmax_agree = float(s.split("=")[-1])
        if s.startswith("kl(teacher||student,last_pos)"):
            ts_kl_last = float(s.split("=")[-1])
        if s.startswith("top") and "_overlap(last_pos)=" in s:
            ts_topk_overlap_last = float(s.split("=")[-1])

    teacher_ce = float("nan")
    student_ce = float("nan")
    if int(loss_window) > 0:
        try:
            teacher_ce = _ce_window_report(lt, y, window=int(loss_window))
            student_ce = _ce_window_report(ls, y, window=int(loss_window))
            logger.info(f"ce(last_{int(loss_window)} tokens) teacher={teacher_ce:.4f} student={student_ce:.4f}")
        except Exception as e:
            logger.warning(f"ce window compute failed: {_fmt_exc(e)}")

    # Gate diagnostics
    gates = _gate_report(student)
    if gates:
        logger.header("DBA gate saturation (student)")
        for line in gates[:32]:
            logger.info(line)
        if len(gates) > 32:
            logger.info(f"... ({len(gates) - 32} more layers)")
    else:
        logger.info("No decoupled_gate_logit parameters found on student.")
        if any("decoupled_gate_logit" in k for k in student_sd.keys()):
            logger.warning("Checkpoint contains decoupled_gate_logit keys, but the built model has no gate parameters.")
        else:
            logger.info("Checkpoint also contains no decoupled_gate_logit keys (gate likely disabled in the run).")

    attn_lines = _attention_param_quickcheck(student, max_layers=6)
    if attn_lines:
        logger.header("Student attention quickcheck", "first few attention modules")
        for l in attn_lines:
            logger.info(l)

    # Per-layer mismatch (attention outputs)
    logger.header("Layer mismatch (attention outputs)", "teacher vs student")
    outs_t = _trace_attention_outputs(teacher, x)
    outs_s = _trace_attention_outputs(student, x)
    n = min(len(outs_t), len(outs_s))
    if n == 0:
        logger.warning("No attention outputs traced (predicate mismatch?)")
        return {
            "stage": stage,
            "teacher_ckpt": teacher_ckpt,
            "student_ckpt": student_ckpt,
            "teacher_logits": lt,
            "student_logits": ls,
            "teacher_entropy": teacher_entropy,
            "student_entropy": student_entropy,
            "teacher_ce": teacher_ce,
            "student_ce": student_ce,
            "ts_argmax_agree": ts_argmax_agree,
            "ts_kl_last": ts_kl_last,
            "ts_topk_overlap_last": ts_topk_overlap_last,
            "layer_scores": [],
        }

    scores: list[tuple[int, float, float]] = []
    for i in range(n):
        a = outs_t[i].detach().float()
        b = outs_s[i].detach().float()
        if a.shape != b.shape:
            continue
        l1 = float((a - b).abs().mean().item())
        l2 = float(((a - b) ** 2).mean().sqrt().item())
        scores.append((i, l1, l2))

    scores.sort(key=lambda t: t[2], reverse=True)
    topk = max(1, int(topk_layers))
    for i, l1, l2 in scores[:topk]:
        logger.info(f"attn_layer{i:02d}: mean|Δ|={l1:.6f} rms|Δ|={l2:.6f}")

    return {
        "stage": stage,
        "teacher_ckpt": teacher_ckpt,
        "student_ckpt": student_ckpt,
        "student_sig": sig,
        "teacher_logits": lt,
        "student_logits": ls,
        "teacher_entropy": teacher_entropy,
        "student_entropy": student_entropy,
        "teacher_ce": teacher_ce,
        "student_ce": student_ce,
        "ts_argmax_agree": ts_argmax_agree,
        "ts_kl_last": ts_kl_last,
        "ts_topk_overlap_last": ts_topk_overlap_last,
        "layer_scores": scores,
    }


def _attention_param_quickcheck(student: nn.Module, *, max_layers: int = 6) -> list[str]:
    """Surface a few attention-module internals without assuming exact class imports."""
    lines: list[str] = []
    idx = 0
    for name, m in student.named_modules():
        if idx >= int(max_layers):
            break
        if not (hasattr(m, "out_proj") and (hasattr(m, "qkv_proj") or hasattr(m, "q_sem") or hasattr(m, "q_geo"))):
            continue

        parts: list[str] = [f"{name}"]
        g = getattr(m, "decoupled_gate_logit", None)
        if isinstance(g, torch.nn.Parameter):
            gv = torch.sigmoid(g.detach().float())
            parts.append(f"gate(sigmoid) mean={float(gv.mean().item()):.3f} min={float(gv.min().item()):.3f} max={float(gv.max().item()):.3f}")
        if hasattr(m, "q_sem") and hasattr(m, "q_geo"):
            qs = getattr(m, "q_sem", None)
            qg = getattr(m, "q_geo", None)
            if hasattr(qs, "weight") and hasattr(qg, "weight"):
                ws = getattr(qs, "weight", None)
                wg = getattr(qg, "weight", None)
                if isinstance(ws, Tensor) and isinstance(wg, Tensor):
                    parts.append(f"q_sem_shape={tuple(ws.shape)} q_geo_shape={tuple(wg.shape)}")
        op = getattr(m, "out_proj", None)
        if hasattr(op, "weight"):
            w = getattr(op, "weight", None)
            if isinstance(w, Tensor):
                parts.append(f"out_proj||w||={float(w.detach().float().norm().item()):.4g}")

        lines.append(" • ".join(parts))
        idx += 1
    return lines


def _gate_report(student: nn.Module) -> list[str]:
    rows: list[str] = []
    idx = 0
    for m in student.modules():
        g = getattr(m, "decoupled_gate_logit", None)
        if not isinstance(g, torch.nn.Parameter):
            continue
        gv = torch.sigmoid(g.detach().float())
        rows.append(
            f"layer{idx:02d}: gate sigmoid min={float(gv.min().item()):.3f} "
            f"mean={float(gv.mean().item()):.3f} max={float(gv.max().item()):.3f} "
            f"p<0.05={float((gv<0.05).float().mean().item()):.3f} p>0.95={float((gv>0.95).float().mean().item()):.3f}"
        )
        idx += 1
    return rows


def _attention_trace_predicate(_name: str, m: nn.Module) -> bool:
    # Avoid isinstance() issues due to dual module roots; use structural checks.
    n = type(m).__name__
    if n in {"AttentionLayer", "StandardAttentionLayer", "DecoupledAttentionLayer"}:
        return True
    return hasattr(m, "out_proj") and (hasattr(m, "qkv_proj") or (hasattr(m, "q_sem") and hasattr(m, "q_geo")))


def _trace_attention_outputs(model: nn.Module, x: Tensor) -> list[Tensor]:
    tr = Trace(model, predicate=_attention_trace_predicate)
    tr.clear()
    with torch.inference_mode():
        with tr:
            try:
                _ = model(x)
            except TraceStop:
                pass
    return list(tr.outputs)


def _grad_norm_report(student: nn.Module) -> list[str]:
    """Collect grad norms for DBA-relevant params after backward()."""
    lines: list[str] = []
    buckets = {
        "gate": ("decoupled_gate_logit", "decoupled_gate_proj"),
        "qk_semgeo": ("q_sem", "k_sem", "q_geo", "k_geo"),
        "vo": ("v_proj", "out_proj"),
        "mlp": ("w_gate_up", "w_down"),
        "embed": ("embedder", "token_embedding"),
        "norm": ("RMSNorm", "LayerNorm", "norm"),
    }

    def add_bucket(name: str, keys: tuple[str, ...]) -> None:
        total = 0.0
        maxv = 0.0
        count = 0
        for pn, p in student.named_parameters():
            if p.grad is None:
                continue
            if any(k in pn for k in keys):
                g = p.grad.detach()
                if g.numel() == 0:
                    continue
                n = float(g.float().norm().item())
                total += n
                maxv = max(maxv, n)
                count += 1
        lines.append(f"{name}: params_with_grad={count} sum||g||={total:.6g} max||g||={maxv:.6g}")

    for bn, keys in buckets.items():
        add_bucket(bn, keys)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-ckpt", default="hf://meta-llama/Llama-3.2-1B", help="Post-finetune teacher checkpoint.")
    ap.add_argument("--student-ckpt", default="runs/paper/finetune_global_final.pt", help="Post-finetune student checkpoint.")
    ap.add_argument("--teacher-pre-ckpt", default=None, help="Optional pre-finetune teacher checkpoint.")
    ap.add_argument("--student-pre-ckpt", default=None, help="Optional pre-finetune student checkpoint.")
    ap.add_argument("--dataset", default="artifacts/datasets/fineweb_llama/fineweb_llama_1b.npy", help="(Deprecated) Single eval dataset.")
    ap.add_argument(
        "--eval-dataset",
        action="append",
        default=[],
        help="Repeatable: evaluation dataset path (.npy). If omitted, uses --dataset.",
    )
    ap.add_argument(
        "--eval-name",
        action="append",
        default=[],
        help="Repeatable: name for each --eval-dataset (same order). Defaults to dataset basename.",
    )
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--batch-index", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32"])
    ap.add_argument("--unsafe-pickle-load", action="store_true")
    ap.add_argument("--topk-layers", type=int, default=5)
    ap.add_argument("--logits-topk", type=int, default=10, help="Top-k for overlap metrics on last position.")
    ap.add_argument(
        "--loss-window",
        type=int,
        default=256,
        help="Compute CE over last N tokens for teacher/student (0 disables).",
    )
    ap.add_argument("--plot", action="store_true", help="Show a matplotlib dashboard (interactive).")
    ap.add_argument("--plot-out", default=None, help="Save dashboard image to this path (e.g. dashboard.png).")
    ap.add_argument("--plot-dpi", type=int, default=150, help="DPI for --plot-out.")
    ap.add_argument(
        "--check-gate-grads",
        action="store_true",
        help="Run 1 backward pass on the batch and report grad norms for gate/qk/v/o/etc.",
    )
    args = ap.parse_args()

    device = torch.device(str(args.device))
    dt = weight_dtype(device, args.dtype if args.dtype != "auto" else "auto")
    logger.header("DBA Diagnose", f"device={device.type} dtype={dt}")

    eval_datasets = list(cast(list[str], args.eval_dataset))
    eval_names = list(cast(list[str], args.eval_name))
    if not eval_datasets:
        eval_datasets = [str(args.dataset)]
    if eval_names and len(eval_names) != len(eval_datasets):
        logger.warning("--eval-name count does not match --eval-dataset count; ignoring --eval-name.")
        eval_names = []

    eval_specs: list[tuple[str, str]] = []
    for i, p in enumerate(eval_datasets):
        name = eval_names[i] if eval_names else Path(p).name.replace(".npy", "").replace(".npz", "")
        eval_specs.append((name, p))

    pre_enabled = bool(args.teacher_pre_ckpt) and bool(args.student_pre_ckpt)
    if (bool(args.teacher_pre_ckpt) or bool(args.student_pre_ckpt)) and not pre_enabled:
        logger.warning("Pre-finetune comparison requested, but both --teacher-pre-ckpt and --student-pre-ckpt are required.")

    for dataset_name, dataset_path in eval_specs:
        logger.header("Evaluation", f"{dataset_name} ({dataset_path})")

        ds = NpyDataset(str(dataset_path), block_size=int(args.block_size))
        item = ds[int(args.batch_index)]
        x = cast(Tensor, item["input_ids"]).unsqueeze(0).to(device=device)
        y = cast(Tensor, item["target_ids"]).unsqueeze(0).to(device=device)
        mx = int(max(x.max().item(), y.max().item()))
        logger.info(f"batch token max_id={mx} block_size={int(args.block_size)}")

        pre_res: dict[str, Any] | None = None
        post_res: dict[str, Any] | None = None
        ft_delta: dict[str, Any] | None = None

        if pre_enabled:
            pre_res = _run_pair(
                stage="pre-ft",
                teacher_ckpt=str(args.teacher_pre_ckpt),
                student_ckpt=str(args.student_pre_ckpt),
                device=device,
                dt=dt,
                x=x,
                y=y,
                topk_layers=int(args.topk_layers),
                logits_topk=int(args.logits_topk),
                loss_window=int(args.loss_window),
                unsafe_pickle_load=bool(args.unsafe_pickle_load),
            )

        post_res = _run_pair(
            stage="post-ft",
            teacher_ckpt=str(args.teacher_ckpt),
            student_ckpt=str(args.student_ckpt),
            device=device,
            dt=dt,
            x=x,
            y=y,
            topk_layers=int(args.topk_layers),
            logits_topk=int(args.logits_topk),
            loss_window=int(args.loss_window),
            unsafe_pickle_load=bool(args.unsafe_pickle_load),
        )

        if pre_enabled and pre_res is not None and post_res is not None:
            pre_teacher_logits = cast(Tensor, pre_res["teacher_logits"])
            pre_student_logits = cast(Tensor, pre_res["student_logits"])
            post_teacher_logits = cast(Tensor, post_res["teacher_logits"])
            post_student_logits = cast(Tensor, post_res["student_logits"])

            logger.header(f"Finetune delta (pre → post) • {dataset_name}", "logits agreement")
            for line in _pairwise_lastpos_report(
                pre_teacher_logits, post_teacher_logits, label="teacher(pre,post)", topk=int(args.logits_topk)
            ):
                logger.info(line)
            for line in _pairwise_lastpos_report(
                pre_student_logits, post_student_logits, label="student(pre,post)", topk=int(args.logits_topk)
            ):
                logger.info(line)

            if int(args.loss_window) > 0:
                tce_pre = float(pre_res.get("teacher_ce", float("nan")))
                tce_post = float(post_res.get("teacher_ce", float("nan")))
                sce_pre = float(pre_res.get("student_ce", float("nan")))
                sce_post = float(post_res.get("student_ce", float("nan")))
                logger.info(f"teacher ΔCE(pre→post, last_{int(args.loss_window)})={tce_post - tce_pre:+.4f}")
                logger.info(f"student ΔCE(pre→post, last_{int(args.loss_window)})={sce_post - sce_pre:+.4f}")

            kl_ts_pre = float(pre_res.get("ts_kl_last", float("nan")))
            kl_ts_post = float(post_res.get("ts_kl_last", float("nan")))
            agree_ts_pre = float(pre_res.get("ts_argmax_agree", float("nan")))
            agree_ts_post = float(post_res.get("ts_argmax_agree", float("nan")))
            logger.info(f"ΔKL(teacher||student,last_pos) pre→post = {kl_ts_post - kl_ts_pre:+.6f}")
            logger.info(f"Δargmax_agreement(teacher,student) pre→post = {agree_ts_post - agree_ts_pre:+.4f}")

            def kl_pre_post(a: Tensor, b: Tensor) -> float:
                a_last = a.detach().float()[:, -1, :]
                b_last = b.detach().float()[:, -1, :]
                logpa = torch.log_softmax(a_last, dim=-1)
                logpb = torch.log_softmax(b_last, dim=-1)
                pa = torch.softmax(a_last, dim=-1)
                return float((pa * (logpa - logpb)).sum(dim=-1).mean().item())

            ft_delta = {
                "teacher_dce": f"{(float(post_res.get('teacher_ce', float('nan'))) - float(pre_res.get('teacher_ce', float('nan')))):+.4f}",
                "student_dce": f"{(float(post_res.get('student_ce', float('nan'))) - float(pre_res.get('student_ce', float('nan')))):+.4f}",
                "teacher_kl_pre_post": f"{kl_pre_post(pre_teacher_logits, post_teacher_logits):.6f}",
                "student_kl_pre_post": f"{kl_pre_post(pre_student_logits, post_student_logits):.6f}",
                "d_kl_ts": f"{(kl_ts_post - kl_ts_pre):+.6f}",
                "d_agree_ts": f"{(agree_ts_post - agree_ts_pre):+.4f}",
            }

            sig_pre = cast(dict[str, Any], pre_res.get("student_sig", {}) or {})
            sig_post = cast(dict[str, Any], post_res.get("student_sig", {}) or {})
            keys = ("vocab_size", "d_model", "n_layers", "sem_dim", "geo_dim", "attn_dim", "decoupled_gate")
            diffs = []
            for k in keys:
                if sig_pre.get(k) != sig_post.get(k):
                    diffs.append(f"{k}: pre={sig_pre.get(k)} post={sig_post.get(k)}")
            if diffs:
                logger.warning("Pre/post student config signatures differ (comparisons may be misleading): " + "; ".join(diffs))

        # Dashboard output path (auto-suffix when multiple datasets).
        plot_out = str(args.plot_out) if args.plot_out else None
        if plot_out and len(eval_specs) > 1:
            p = Path(plot_out)
            plot_out = str(p.with_name(f"{p.stem}_{dataset_name}{p.suffix}"))

        _maybe_dashboard(
            dataset_name=dataset_name,
            dataset_path=str(dataset_path),
            pre=pre_res,
            post=cast(dict[str, Any], post_res),
            ft_delta=ft_delta,
            plot=bool(args.plot),
            plot_out=plot_out,
            plot_dpi=int(args.plot_dpi),
        )

        # Try to free memory between datasets on MPS.
        try:
            del ds, item, x, y, pre_res, post_res
            if device.type == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    if bool(args.check_gate_grads):
        logger.header("Gradient norms (1 backward pass)", "student")
        logger.warning(
            "--check-gate-grads is currently stage-agnostic in this script. "
            "If you want grads, run a separate forward/backward in your main training codepath."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

