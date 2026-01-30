from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from benchmark.attention_multi_model import render_multi_model_attention_comparison
from benchmark.behavior.types import GeneratedCase
from console import logger
from data.tokenizers.builder import TokenizerBuilder
from instrumentation.viz import TrainingVizContext
# IMPORTANT: model configs build layers from the `caramba.layer` namespace
# (see `config.layer.LayerType.module_name()`), so we must import from the same
# namespace here. Importing `layer.attention` would create a different module,
# causing `isinstance(mod, AttentionLayer)` to fail and resulting in empty viz events.
from caramba.layer.attention import AttentionLayer


class ModelWithCtx(Protocol):
    def __call__(self, x: torch.Tensor, ctx: Any = None) -> Any: ...


def dump_attention_multi_model(
    *,
    models: dict[str, nn.Module],
    cases: list[GeneratedCase],
    benchmark_id: str,
    output_dir: Path,
    device: torch.device,
    tokenizer_config,
    max_tokens: int | None,
    max_heads: int | None,
    anchor: str | None,
) -> dict[str, Path]:
    """Dump attention patterns and required visualizations for N models.

    Strict behavior:
    - If a case cannot be tokenized, or target span cannot be located, we raise.
    - If a model does not support ctx-enabled forward, we raise.
    - If plotting fails, we raise.
    """
    if not models:
        raise ValueError("dump_attention_multi_model: no models provided.")
    if not cases:
        raise ValueError("dump_attention_multi_model: no cases provided.")

    out_base = Path(output_dir)
    safe_benchmark_id = str(benchmark_id).replace("/", "_").replace("\\", "_")
    out_dir = out_base / "attention_dump" / safe_benchmark_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = TokenizerBuilder().build(tokenizer_config)

    tok_lim = int(max_tokens) if (max_tokens is not None and int(max_tokens) > 0) else None
    head_lim = int(max_heads) if (max_heads is not None and int(max_heads) > 0) else None
    anchor_s = str(anchor) if anchor else ""

    artifacts: dict[str, Path] = {}

    for case in cases:
        case_dir = out_dir / str(case.id)
        case_dir.mkdir(parents=True, exist_ok=True)

        prompt_ids = tok.encode(str(case.prompt))
        if not prompt_ids:
            raise RuntimeError(f"Attention dump: case {case.id!r} tokenized to empty prompt.")

        limit = int(min(len(prompt_ids), int(tok_lim))) if tok_lim is not None else int(len(prompt_ids))
        prompt_ids = list(prompt_ids[:limit])
        token_strs = [tok.decode([i]).replace("\n", "\\n") for i in prompt_ids]

        split_idx = _find_anchor_token_index(token_strs, target=anchor_s) if anchor_s else None
        split = int(split_idx if split_idx is not None else len(token_strs) // 2)

        # Target span is REQUIRED for attention dump visuals.
        target_text = str(case.target_text) if case.target_text is not None else str(case.expected)
        target_span = _find_text_span_in_prompt(
            tok=tok,
            prompt_ids=prompt_ids,
            text=str(target_text),
        )
        if target_span is None:
            # This commonly happens when:
            # - prompts are truncated via max_tokens
            # - target_text includes quoting/formatting that doesn't round-trip through token IDs
            # Attention visualization is auxiliary; don't fail the whole benchmark run.
            logger.warning(
                f"Attention dump: skipping case {case.id!r}: target_text not found in prompt tokens "
                f"(target_text={target_text!r})."
            )
            continue
        a0, a1 = target_span

        # Case-level JSON for auditability + downstream analysis.
        case_json = {
            "case_id": str(case.id),
            "category": str(case.category),
            "difficulty": str(case.difficulty.value),
            "kind": str(case.kind.value),
            "prompt": str(case.prompt),
            "expected": str(case.expected),
            "target_text": str(target_text),
            "target_span": [int(a0), int(a1)],
            "target_tokens": token_strs[int(a0) : int(a1) + 1],
            "anchor": str(anchor_s),
            "split": int(split),
            "models": [str(k) for k in models.keys()],
            "metadata": case.metadata,
        }
        (case_dir / "case.json").write_text(json.dumps(case_json, indent=2), encoding="utf-8")
        (case_dir / "tokens.json").write_text(json.dumps(token_strs, indent=2), encoding="utf-8")

        # Run all models and collect attention events
        model_events: dict[str, dict[str, Any]] = {}
        for model_name, model in models.items():
            model_dir = case_dir / str(model_name)
            model_dir.mkdir(parents=True, exist_ok=True)

            _assign_attention_viz_ids(model)
            ctx = TrainingVizContext(
                enabled=True,
                step=0,
                max_tokens=int(limit),
                max_heads=int(head_lim) if head_lim is not None else 1_000_000,
            )
            x = torch.tensor([prompt_ids], device=device, dtype=torch.long)
            with torch.no_grad():
                m_ctx = cast(ModelWithCtx, model)
                _ = m_ctx(x, ctx=ctx)

            event = ctx.to_event()
            model_events[str(model_name)] = event
            (model_dir / "attn.json").write_text(json.dumps(event, indent=2), encoding="utf-8")

            summary = attention_mass_summary(
                event=event,
                tokens=token_strs,
                split=int(split),
                target_span=(int(a0), int(a1)),
            )
            (model_dir / "mass.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

            _render_attention_pngs(
                model_dir=model_dir,
                event=event,
                tokens=token_strs,
                split=int(split),
                target_span=(int(a0), int(a1)),
                case_id=str(case.id),
                model_tag=str(model_name),
            )

        # Aggregated comparison overview (same 3 visualizations across models).
        comp = render_multi_model_attention_comparison(
            model_events=model_events,
            tokens=token_strs,
            split=int(split),
            case_id=str(case.id),
            output_dir=case_dir,
            answer_span=(int(a0), int(a1)),
            max_heads=int(head_lim) if head_lim is not None else 4,
        )
        for name, path in comp.items():
            artifacts[f"{case.id}_{name}"] = path

    return artifacts


def dump_attention_multi_model_isolated(
    *,
    model_names: list[str],
    load_model,
    unload_model,
    cases: list[GeneratedCase],
    benchmark_id: str,
    output_dir: Path,
    device: torch.device,
    tokenizer_config,
    max_tokens: int | None,
    max_heads: int | None,
    anchor: str | None,
) -> dict[str, Path]:
    """Dump attention patterns and required visualizations for N models (isolated load/unload).

    This is the isolated equivalent of `dump_attention_multi_model`, used when the
    benchmark runner is in process isolation mode and can only keep one model in memory.

    Strict behavior:
    - If a case cannot be tokenized, or target span cannot be located, we raise.
    - If a model does not support ctx-enabled forward, we raise.
    - If plotting fails, we raise.
    """
    model_names = [str(n) for n in model_names if str(n).strip()]
    if not model_names:
        raise ValueError("dump_attention_multi_model_isolated: no model_names provided.")
    if not cases:
        raise ValueError("dump_attention_multi_model_isolated: no cases provided.")

    out_base = Path(output_dir)
    safe_benchmark_id = str(benchmark_id).replace("/", "_").replace("\\", "_")
    out_dir = out_base / "attention_dump" / safe_benchmark_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = TokenizerBuilder().build(tokenizer_config)

    tok_lim = int(max_tokens) if (max_tokens is not None and int(max_tokens) > 0) else None
    head_lim = int(max_heads) if (max_heads is not None and int(max_heads) > 0) else None
    anchor_s = str(anchor) if anchor else ""

    artifacts: dict[str, Path] = {}

    for case in cases:
        case_dir = out_dir / str(case.id)
        case_dir.mkdir(parents=True, exist_ok=True)

        prompt_ids = tok.encode(str(case.prompt))
        if not prompt_ids:
            raise RuntimeError(f"Attention dump: case {case.id!r} tokenized to empty prompt.")

        limit = int(min(len(prompt_ids), int(tok_lim))) if tok_lim is not None else int(len(prompt_ids))
        prompt_ids = list(prompt_ids[:limit])
        token_strs = [tok.decode([i]).replace("\n", "\\n") for i in prompt_ids]

        split_idx = _find_anchor_token_index(token_strs, target=anchor_s) if anchor_s else None
        split = int(split_idx if split_idx is not None else len(token_strs) // 2)

        target_text = str(case.target_text) if case.target_text is not None else str(case.expected)
        target_span = _find_text_span_in_prompt(tok=tok, prompt_ids=prompt_ids, text=str(target_text))
        if target_span is None:
            logger.warning(
                f"Attention dump: skipping case {case.id!r}: target_text not found in prompt tokens "
                f"(target_text={target_text!r})."
            )
            continue
        a0, a1 = target_span

        case_json = {
            "case_id": str(case.id),
            "category": str(case.category),
            "difficulty": str(case.difficulty.value),
            "kind": str(case.kind.value),
            "prompt": str(case.prompt),
            "expected": str(case.expected),
            "target_text": str(target_text),
            "target_span": [int(a0), int(a1)],
            "target_tokens": token_strs[int(a0) : int(a1) + 1],
            "anchor": str(anchor_s),
            "split": int(split),
            "models": [str(n) for n in model_names],
            "metadata": case.metadata,
        }
        (case_dir / "case.json").write_text(json.dumps(case_json, indent=2), encoding="utf-8")
        (case_dir / "tokens.json").write_text(json.dumps(token_strs, indent=2), encoding="utf-8")

        model_events: dict[str, dict[str, Any]] = {}
        for model_name in model_names:
            model = load_model(model_name)
            try:
                model.eval()
                model_dir = case_dir / str(model_name)
                model_dir.mkdir(parents=True, exist_ok=True)

                _assign_attention_viz_ids(model)
                ctx = TrainingVizContext(
                    enabled=True,
                    step=0,
                    max_tokens=int(limit),
                    max_heads=int(head_lim) if head_lim is not None else 1_000_000,
                )
                x = torch.tensor([prompt_ids], device=device, dtype=torch.long)
                with torch.no_grad():
                    m_ctx = cast(ModelWithCtx, model)
                    _ = m_ctx(x, ctx=ctx)

                event = ctx.to_event()
                model_events[str(model_name)] = event
                (model_dir / "attn.json").write_text(json.dumps(event, indent=2), encoding="utf-8")

                summary = attention_mass_summary(
                    event=event,
                    tokens=token_strs,
                    split=int(split),
                    target_span=(int(a0), int(a1)),
                )
                (model_dir / "mass.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

                _render_attention_pngs(
                    model_dir=model_dir,
                    event=event,
                    tokens=token_strs,
                    split=int(split),
                    target_span=(int(a0), int(a1)),
                    case_id=str(case.id),
                    model_tag=str(model_name),
                )
            finally:
                unload_model(model)

        comp = render_multi_model_attention_comparison(
            model_events=model_events,
            tokens=token_strs,
            split=int(split),
            case_id=str(case.id),
            output_dir=case_dir,
            answer_span=(int(a0), int(a1)),
            max_heads=int(head_lim) if head_lim is not None else 4,
        )
        for name, path in comp.items():
            artifacts[f"{case.id}_{name}"] = path

    return artifacts


def _assign_attention_viz_ids(model: nn.Module) -> None:
    """Ensure attention layers have stable viz ids/names for ctx recording."""
    i = 0
    for name, mod in model.named_modules():
        if isinstance(mod, AttentionLayer):
            setattr(mod, "_viz_index", int(i))
            setattr(mod, "_viz_name", str(name))
            i += 1


def _find_anchor_token_index(tokens: list[str], *, target: str) -> int | None:
    if not target:
        return None
    for i, t in enumerate(tokens):
        if target in t:
            return int(i)
    # fallback: accumulated string
    from itertools import accumulate

    for i, s in enumerate(accumulate(tokens)):
        if target in s:
            return int(i)
    return None


def _find_subseq(hay: list[int], needle: list[int]) -> int | None:
    if not needle or not hay or len(needle) > len(hay):
        return None
    last: int | None = None
    for i in range(0, len(hay) - len(needle) + 1):
        if hay[i : i + len(needle)] == needle:
            last = int(i)
    return last


def _find_text_span_in_prompt(*, tok: Any, prompt_ids: list[int], text: str) -> tuple[int, int] | None:
    exp = str(text)
    # Try whitespace variants (space-sensitive tokenizers).
    cands: list[str] = [exp, exp.strip()]
    # Common formatting variants (e.g. JSON strings like "\"lemon\"").
    if len(exp) >= 2 and ((exp[0] == exp[-1] == '"') or (exp[0] == exp[-1] == "'")):
        inner = exp[1:-1]
        if inner:
            cands.append(inner)
            cands.append(inner.strip())
    if exp and not exp.startswith(" "):
        cands.append(" " + exp)
        cands.append(" " + exp.lstrip())
    if exp and exp.startswith(" "):
        cands.append(exp.lstrip())

    for s in cands:
        ids = tok.encode(str(s))
        j = _find_subseq(prompt_ids, list(ids))
        if j is not None and len(ids) > 0:
            return int(j), int(j + len(ids) - 1)

    # Fallback: locate the target as a substring in the decoded token stream and map back
    # to token indices. This is less strict than token-subsequence matching but avoids
    # spurious failures from formatting/quoting differences.
    try:
        pieces = [str(tok.decode([i])) for i in prompt_ids]
        full = "".join(pieces)
        s0 = full.find(exp)
        if s0 < 0 and exp.strip():
            s0 = full.find(exp.strip())
        if s0 < 0:
            # Try inner (quote-stripped) as well.
            if len(exp) >= 2 and ((exp[0] == exp[-1] == '"') or (exp[0] == exp[-1] == "'")):
                inner = exp[1:-1]
                if inner:
                    s0 = full.find(inner)
        if s0 < 0:
            return None
        s1 = s0 + len(exp if full.find(exp) == s0 else (exp.strip() if full.find(exp.strip()) == s0 else exp[1:-1]))
        # Map character span [s0, s1) to token indices using cumulative ends.
        ends: list[int] = []
        cur = 0
        for p in pieces:
            cur += len(p)
            ends.append(cur)
        a0 = next((i for i, e in enumerate(ends) if e > s0), None)
        a1 = next((i for i, e in enumerate(ends) if e >= s1), None)
        if a0 is None or a1 is None:
            return None
        return int(a0), int(a1)
    except Exception:
        return None
    return None


def attention_mass_summary(
    *,
    event: dict[str, Any],
    tokens: list[str],
    split: int,
    target_span: tuple[int, int],
) -> dict[str, Any]:
    """Per-layer summary for the final query token (mean over heads)."""
    a0, a1 = target_span
    if a0 < 0 or a1 < 0 or a1 < a0:
        raise ValueError("attention_mass_summary: invalid target_span.")

    split2 = max(0, min(int(split), len(tokens)))
    anchor_start = max(0, int(split2) - 2)
    anchor_end = min(len(tokens) - 1, int(split2) + 2)

    out: dict[str, Any] = {
        "split": int(split2),
        "anchor_span": [int(anchor_start), int(anchor_end)],
        "anchor_tokens": tokens[int(anchor_start) : int(anchor_end) + 1],
        "target_span": [int(a0), int(a1)],
        "target_tokens": tokens[int(a0) : int(a1) + 1],
        "layers": [],
    }
    layers = event.get("layers", [])
    if not isinstance(layers, list):
        raise TypeError("attention_mass_summary: event['layers'] must be a list.")

    for layer in layers:
        if not isinstance(layer, dict):
            raise TypeError("attention_mass_summary: layer must be a dict.")
        attn = layer.get("attn")
        if not isinstance(attn, dict):
            continue
        mats = attn.get("matrices")
        if not isinstance(mats, list) or not mats:
            continue

        # mats: list[head][tq][tk]
        head_arrays: list[np.ndarray] = []
        for h in mats:
            arr = np.asarray(h, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError("attention_mass_summary: head matrix must be 2D.")
            head_arrays.append(arr)
        if not head_arrays:
            continue
        m = np.stack(head_arrays, axis=0)  # (H,tq,tk)
        if m.shape[1] <= 0 or m.shape[2] <= 0:
            continue
        dist = m[:, -1, :].mean(axis=0)  # (tk,)

        tk = int(dist.shape[0])
        s = max(0, min(int(split2), tk))
        a0c = max(0, min(int(a0), tk - 1))
        a1c = max(0, min(int(a1), tk - 1))
        if a0c > a1c:
            raise ValueError("attention_mass_summary: target span out of bounds.")

        mass_exemplar = float(np.sum(dist[:s]))
        mass_target_region = float(np.sum(dist[s:]))
        mass_anchor = float(np.sum(dist[int(anchor_start) : int(anchor_end) + 1])) if tk > 0 else 0.0
        mass_target = float(np.sum(dist[int(a0c) : int(a1c) + 1]))

        top1_i = int(np.argmax(dist))
        top1_w = float(dist[top1_i])
        top1_tok = tokens[top1_i] if 0 <= top1_i < len(tokens) else ""
        top1_in_target = bool(int(a0c) <= top1_i <= int(a1c))

        out["layers"].append(
            {
                "index": int(layer.get("index", -1)),
                "name": str(layer.get("name", "")),
                "mode": str(layer.get("mode", "")),
                "tq": int(m.shape[1]),
                "tk": int(tk),
                "mass_exemplar": mass_exemplar,
                "mass_target_region": mass_target_region,
                "mass_anchor": mass_anchor,
                "mass_target": mass_target,
                "top1": {
                    "i": top1_i,
                    "w": top1_w,
                    "tok": top1_tok,
                    "in_target": top1_in_target,
                },
                # Full distribution for downstream exact computations.
                "dist": dist.tolist(),
            }
        )

    return out


def _render_attention_pngs(
    *,
    model_dir: Path,
    event: dict[str, Any],
    tokens: list[str],
    split: int,
    target_span: tuple[int, int],
    case_id: str,
    model_tag: str,
) -> None:
    layers = event.get("layers", [])
    if not isinstance(layers, list) or not layers:
        raise RuntimeError("render_attention_pngs: event has no layers.")

    # Collect per-layer last-row (mean over heads) and last-layer heads.
    rows: list[np.ndarray] = []
    last_layer_heads: list[np.ndarray] | None = None
    for layer in layers:
        if not isinstance(layer, dict):
            raise TypeError("render_attention_pngs: layer must be dict.")
        attn = layer.get("attn")
        if not isinstance(attn, dict):
            continue
        mats = attn.get("matrices")
        if not isinstance(mats, list) or not mats:
            continue
        head_arrays: list[np.ndarray] = []
        for h in mats:
            a = np.asarray(h, dtype=np.float32)
            if a.ndim != 2:
                raise ValueError("render_attention_pngs: head matrix must be 2D.")
            head_arrays.append(a)
        if not head_arrays:
            continue
        m = np.stack(head_arrays, axis=0)  # (H,tq,tk)
        last = m[:, -1, :].mean(axis=0)  # (tk,)
        rows.append(last)
        last_layer_heads = head_arrays

    if not rows:
        raise RuntimeError("render_attention_pngs: no usable attention matrices.")

    M = np.stack(rows, axis=0)  # (L,tk)
    L, tk = int(M.shape[0]), int(M.shape[1])
    split2 = max(0, min(int(split), tk))
    gold = "#d4af37"
    a0 = max(0, min(int(target_span[0]), tk - 1))
    a1 = max(0, min(int(target_span[1]), tk - 1))
    if a0 > a1:
        raise ValueError("render_attention_pngs: target span out of bounds.")

    # ---- Style 1: layer × token heatmap (final query token) ----
    fig = plt.figure(figsize=(min(14.0, 0.18 * tk + 4.0), min(10.0, 0.28 * L + 3.0)))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.axvline(split2 - 0.5, color="w", linewidth=1.5, alpha=0.9)
    ax.axvline(a0 - 0.5, color=gold, linewidth=3.0, alpha=0.95)
    ax.axvline(a1 + 0.5, color=gold, linewidth=3.0, alpha=0.95)

    peaks = M.argmax(axis=1)  # (L,)
    ys = np.arange(L, dtype=np.float32)
    xs = peaks.astype(np.float32)
    in_tgt = (xs >= float(a0)) & (xs <= float(a1))
    ax.scatter(xs[~in_tgt], ys[~in_tgt], s=12, c="#ff4d4d", marker="o", edgecolors="black", linewidths=0.3, alpha=0.9, label="peak (off-target)")
    ax.scatter(xs[in_tgt], ys[in_tgt], s=12, c="#2ecc71", marker="o", edgecolors="black", linewidths=0.3, alpha=0.9, label="peak (on-target)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.set_xlabel("key position (prompt tokens)")
    ax.set_ylabel("layer (sampled attention modules)")
    if tk <= 64:
        ax.set_xticks(list(range(tk)))
        ax.set_xticklabels([t if len(t) <= 6 else t[:6] + "…" for t in tokens[:tk]], rotation=90, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="attention weight")
    fig.tight_layout()
    fig.savefig(model_dir / "attn_style1_layer_by_token.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ---- Style 1b: attention mass vs depth ----
    mass_target = M[:, int(a0) : int(a1) + 1].sum(axis=1)
    mass_off = 1.0 - mass_target
    fig = plt.figure(figsize=(10.5, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    mean_t = float(np.mean(mass_target)) if len(mass_target) else 0.0
    ax.plot(mass_target, label=f"mass on target (mean={mean_t:.2f})", linewidth=2.6, color="#2ecc71")
    ax.plot(mass_off, label="mass off target", linewidth=2.0, linestyle="--", color="#ff4d4d", alpha=0.9)
    peak_i = int(np.argmax(mass_target))
    peak_v = float(mass_target[peak_i])
    ax.scatter([peak_i], [peak_v], s=42, c="#2ecc71", edgecolors="black", linewidths=0.4, zorder=5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("layer (sampled attention modules)")
    ax.set_ylabel("attention mass (final query token)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(model_dir / "attn_style1b_mass_by_layer.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ---- Style 2: last-layer per-head attention matrices ----
    if not last_layer_heads:
        raise RuntimeError("render_attention_pngs: no last-layer heads captured.")
    H = int(len(last_layer_heads))
    # Rank heads by target-focus for the final query token.
    scores: list[tuple[float, int]] = []
    for i, arr in enumerate(last_layer_heads):
        row = np.asarray(arr, dtype=np.float32)[-1, :]
        scores.append((float(np.sum(row[int(a0) : int(a1) + 1])), int(i)))
    scores.sort(key=lambda t: t[0], reverse=True)
    order = [i for _, i in scores]

    ncols = min(4, H)
    nrows = int((H + ncols - 1) // ncols)
    fig = plt.figure(figsize=(min(14.0, 3.2 * ncols), min(10.0, 2.8 * nrows)))
    for j, i in enumerate(order):
        arr = np.asarray(last_layer_heads[i], dtype=np.float32)
        ax = fig.add_subplot(nrows, ncols, j + 1)
        ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=1.0)
        ax.axvline(split2 - 0.5, color="w", linewidth=1.0, alpha=0.9)
        ax.axvline(int(a0) - 0.5, color=gold, linewidth=2.4, alpha=0.95)
        ax.axvline(int(a1) + 0.5, color=gold, linewidth=2.4, alpha=0.95)
        # Dot at peak for final query token.
        row = arr[-1, :]
        pk = int(np.argmax(row))
        in_ans = bool(int(a0) <= pk <= int(a1))
        c = "#2ecc71" if in_ans else "#ff4d4d"
        ax.scatter([pk], [arr.shape[0] - 1], s=26, c=c, marker="o", edgecolors="black", linewidths=0.35, alpha=0.9, zorder=6)
        tok_s = tokens[pk] if 0 <= pk < len(tokens) else ""
        tok_s = tok_s if len(tok_s) <= 6 else tok_s[:6] + "…"
        ans_mass = float(np.sum(row[int(a0) : int(a1) + 1]))
        ax.set_title(f"h{i} • tgt={ans_mass:.2f} • pk={tok_s}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(model_dir / "attn_style2_last_layer_heads.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

