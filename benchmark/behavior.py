"""Behavior benchmark: prompt-suite checks for teacher/student.

This is a lightweight "unit test" style benchmark that runs a curated set of
prompt cases with ground-truth evaluation logic and produces:
- per-case outcomes (teacher_ok/student_ok + extracted answers)
- aggregate accuracies
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
import shutil

from caramba.config.benchmark import BehaviorBenchmarkConfig
from caramba.config.eval import EvalCase, EvalThresholds, EvalVerifyConfig
from caramba.console import logger
from caramba.eval.suite import run_eval_verify
from caramba.data.tokenizers.builder import TokenizerBuilder
from caramba.instrumentation.viz import TrainingVizContext
from caramba.layer.attention import AttentionLayer

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

@dataclass
class BehaviorMeasurement:
    case_id: str
    teacher_ok: bool
    student_ok: bool
    teacher_answer: str
    student_answer: str


@dataclass
class BehaviorResult:
    """Behavior benchmark results for a teacher/student pair."""

    benchmark_id: str
    measurements: list[BehaviorMeasurement] = field(default_factory=list)

    @property
    def teacher_accuracy(self) -> float:
        if not self.measurements:
            return 0.0
        return sum(1 for m in self.measurements if m.teacher_ok) / float(len(self.measurements))

    @property
    def student_accuracy(self) -> float:
        if not self.measurements:
            return 0.0
        return sum(1 for m in self.measurements if m.student_ok) / float(len(self.measurements))


class BehaviorBenchmark:
    def __init__(self, config: BehaviorBenchmarkConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

    def _load_cases(self) -> list[EvalCase]:
        path = Path(self.config.cases_file)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("BehaviorBenchmark cases_file must be a non-empty YAML list")
        cases: list[EvalCase] = []
        for i, item in enumerate(payload):
            if not isinstance(item, dict):
                raise TypeError(
                    f"BehaviorBenchmark cases must be dict objects, got {type(item)!r} at index {i}"
                )
            cases.append(EvalCase.model_validate(item))
        return cases

    def run(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        benchmark_id: str,
        output_dir: Path | None = None,
    ) -> BehaviorResult:
        stream_live = bool(getattr(self.config, "stream_live", False))
        max_chars = int(getattr(self.config, "print_max_chars", 160))
        log_file = getattr(self.config, "log_file", None)

        def _truncate(s: str, mx: int = max_chars) -> str:
            ss = str(s).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
            return ss if len(ss) <= mx else ss[: max(0, mx - 1)] + "…"

        # Leverage the existing eval suite implementation.
        cfg = EvalVerifyConfig(
            tokenizer=self.config.tokenizer,
            max_new_tokens=int(self.config.max_new_tokens),
            context_window=self.config.context_window,
            # thresholds are irrelevant for benchmarking (we always record), but required by config
            thresholds=EvalThresholds(min_student_accuracy=0.0, max_accuracy_drop=1.0),
            cases=self._load_cases(),
        )
        # Keep cases for prompt lookup during streaming
        cases = cfg.cases
        case_by_id = {str(c.id): c for c in cases}

        summary = run_eval_verify(
            teacher=teacher,
            student=student,
            cfg=cfg,
            device=self.device,
        )
        out = BehaviorResult(benchmark_id=str(benchmark_id))

        if stream_live:
            logger.console.print()
            logger.console.print(f"[highlight]━━━ Behavior Benchmark: {benchmark_id} ━━━[/highlight]")
            logger.console.print()

        # Prepare log file if configured
        log_lines: list[str] = []
        if log_file:
            log_lines.append(f"{'=' * 80}")
            log_lines.append(f"BEHAVIOR BENCHMARK: {benchmark_id}")
            log_lines.append(f"{'=' * 80}")
            log_lines.append("")

        for i, r in enumerate(summary.results):
            m = BehaviorMeasurement(
                case_id=str(r.case_id),
                teacher_ok=bool(r.teacher_ok),
                student_ok=bool(r.student_ok),
                teacher_answer=str(r.teacher_answer),
                student_answer=str(r.student_answer),
            )
            out.measurements.append(m)

            # Get the original prompt and expected answer
            case = case_by_id.get(m.case_id)
            prompt_full = case.prompt if case else "[prompt not found]"
            expected_full = str(case.answer) if case else "?"

            # Live streaming: print each case as it's processed (truncated)
            if stream_live:
                t_status = "[success]✓[/success]" if m.teacher_ok else "[error]✗[/error]"
                s_status = "[success]✓[/success]" if m.student_ok else "[error]✗[/error]"

                logger.console.print(
                    f"  [{i + 1:>2}/{len(summary.results)}] [highlight]{m.case_id}[/highlight]"
                )
                logger.console.print(
                    f"       [muted]prompt:[/muted] {_truncate(prompt_full, 120)}"
                )
                logger.console.print(
                    f"       [muted]expected:[/muted] {_truncate(expected_full, 80)}"
                )
                logger.console.print(
                    f"       {t_status} [muted]teacher:[/muted] {_truncate(m.teacher_answer, 80)}"
                )
                logger.console.print(
                    f"       {s_status} [muted]student:[/muted] {_truncate(m.student_answer, 80)}"
                )
                logger.console.print()

            # Log file: write full untruncated details
            if log_file:
                t_mark = "✓" if m.teacher_ok else "✗"
                s_mark = "✓" if m.student_ok else "✗"
                log_lines.append(f"[{i + 1}/{len(summary.results)}] {m.case_id}")
                log_lines.append("-" * 40)
                log_lines.append("PROMPT:")
                log_lines.append(prompt_full)
                log_lines.append("")
                log_lines.append(f"EXPECTED: {expected_full}")
                log_lines.append("")
                log_lines.append(f"TEACHER {t_mark}:")
                log_lines.append(m.teacher_answer)
                log_lines.append("")
                log_lines.append(f"STUDENT {s_mark}:")
                log_lines.append(m.student_answer)
                log_lines.append("")
                log_lines.append("")

        # Optional attention dump for selected cases (paper/debug).
        if bool(getattr(self.config, "dump_attention", False)):
            try:
                self._dump_attention(
                    teacher=teacher,
                    student=student,
                    cases=cases,
                    measurements=out.measurements,
                    benchmark_id=str(benchmark_id),
                    output_dir=output_dir,
                )
            except Exception as e:
                logger.warning(f"Attention dump failed (continuing): {e!r}")

        # Print summary
        t_correct = sum(1 for m in out.measurements if m.teacher_ok)
        s_correct = sum(1 for m in out.measurements if m.student_ok)
        total = len(out.measurements)

        if stream_live:
            logger.console.print(
                f"[highlight]━━━ Summary:[/highlight] "
                f"teacher [metric]{t_correct}[/metric]/[metric]{total}[/metric] "
                f"([success]{t_correct / total * 100:.1f}%[/success]) │ "
                f"student [metric]{s_correct}[/metric]/[metric]{total}[/metric] "
                f"([success]{s_correct / total * 100:.1f}%[/success])"
            )
            logger.console.print()

        # Write log file
        if log_file and log_lines:
            log_lines.append(f"{'=' * 80}")
            log_lines.append(f"SUMMARY: teacher {t_correct}/{total} ({t_correct/total*100:.1f}%) │ student {s_correct}/{total} ({s_correct/total*100:.1f}%)")
            log_lines.append(f"{'=' * 80}")

            log_path = Path(log_file)
            if output_dir and not log_path.is_absolute():
                log_path = output_dir / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(log_lines), encoding="utf-8")
            logger.info(f"Full behavior log written to: {log_path}")

        # Optional: print per-case outputs for insight/debugging.
        if bool(getattr(self.config, "print_outputs", False)) and out.measurements:
            max_chars = int(getattr(self.config, "print_max_chars", 160))
            only_fail = bool(getattr(self.config, "print_only_failures", True))

            def _trunc(s: str) -> str:
                ss = str(s).replace("\r\n", "\n").replace("\r", "\n")
                ss = ss.replace("\n", "\\n")
                if len(ss) <= max_chars:
                    return ss
                return ss[: max(0, max_chars - 1)] + "…"

            rows: list[list[str]] = []
            for m in out.measurements:
                disagree = bool(m.teacher_ok) != bool(m.student_ok)
                any_wrong = (not bool(m.teacher_ok)) or (not bool(m.student_ok))
                if only_fail and not (disagree or any_wrong):
                    continue
                rows.append(
                    [
                        str(m.case_id),
                        "✓" if bool(m.teacher_ok) else "✗",
                        "✓" if bool(m.student_ok) else "✗",
                        _trunc(m.teacher_answer),
                        _trunc(m.student_answer),
                    ]
                )
            if rows:
                logger.table(
                    title=f"Behavior outputs • {benchmark_id}",
                    columns=["case", "T", "S", "teacher_output", "student_output"],
                    rows=rows,
                )

        return out

    def _dump_attention(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        cases: list[EvalCase],
        measurements: list[BehaviorMeasurement],
        benchmark_id: str,
        output_dir: Path | None,
    ) -> None:
        out_base = Path(output_dir) if output_dir is not None else Path(".")
        out_dir = out_base / "attention_dump" / str(benchmark_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        case_by_id = {str(c.id): c for c in cases}
        meas_by_id = {str(m.case_id): m for m in measurements}

        # Decide which cases to dump.
        requested = getattr(self.config, "dump_attention_case_ids", None)
        if requested:
            case_ids = [str(x) for x in requested]
        else:
            # Default: dump only failures/disagreements (high-signal).
            case_ids = []
            for m in measurements:
                disagree = bool(m.teacher_ok) != bool(m.student_ok)
                any_wrong = (not bool(m.teacher_ok)) or (not bool(m.student_ok))
                if disagree or any_wrong:
                    case_ids.append(str(m.case_id))

        if not case_ids:
            logger.info("Attention dump enabled, but no matching cases selected.")
            return

        tok = TokenizerBuilder().build(self.config.tokenizer)
        max_tokens = int(getattr(self.config, "dump_attention_max_tokens", 96))
        max_heads = int(getattr(self.config, "dump_attention_max_heads", 4))
        anchor = str(getattr(self.config, "dump_attention_anchor", "A7"))

        for cid in case_ids:
            case = case_by_id.get(cid)
            if case is None:
                continue
            meas = meas_by_id.get(cid)
            case_dir = out_dir / str(cid)
            case_dir.mkdir(parents=True, exist_ok=True)

            # Save raw prompt/expected/outcomes for traceability.
            meta = {
                "case_id": str(cid),
                "kind": str(getattr(case, "kind", "")),
                "prompt": str(case.prompt),
                "expected": case.answer,
            }
            if meas is not None:
                meta["teacher_ok"] = bool(meas.teacher_ok)
                meta["student_ok"] = bool(meas.student_ok)
                meta["teacher_answer"] = str(meas.teacher_answer)
                meta["student_answer"] = str(meas.student_answer)
            (case_dir / "case.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            prompt_ids = tok.encode(case.prompt)
            if not prompt_ids:
                continue

            # Per-token strings for axis labels / debugging.
            token_strs = [tok.decode([i]) for i in prompt_ids[:max_tokens]]
            # Normalize newlines for readability in JSON consumers.
            token_strs = [s.replace("\n", "\\n") for s in token_strs]
            (case_dir / "tokens.json").write_text(json.dumps(token_strs, indent=2), encoding="utf-8")

            # Pick a split index so we can report "exemplar vs target" attention mass.
            split_idx = _find_anchor_token_index(token_strs, anchor=anchor)
            split = int(split_idx if split_idx is not None else len(token_strs) // 2)

            # Run both models.
            for tag, model in (("teacher", teacher), ("student", student)):
                model_dir = case_dir / tag
                model_dir.mkdir(parents=True, exist_ok=True)

                _assign_attention_viz_ids(model)
                ctx = TrainingVizContext(
                    enabled=True,
                    step=0,
                    max_tokens=int(max_tokens),
                    max_heads=int(max_heads),
                )
                x = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
                with torch.no_grad():
                    _ = model(x, ctx=ctx)  # type: ignore[call-arg]

                event = ctx.to_event()
                (model_dir / "attn.json").write_text(json.dumps(event, indent=2), encoding="utf-8")

                # Also write a compact numeric summary per layer:
                # how much the *final* query attends to exemplar region vs target region.
                summary = _attention_mass_summary(event, split=int(split), tokens=token_strs)
                (model_dir / "mass.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

                # Render paper-friendly PNGs (two styles).
                _render_attention_pngs(
                    model_dir=model_dir,
                    event=event,
                    tokens=token_strs,
                    split=int(split),
                    case_id=str(cid),
                    model_tag=str(tag),
                )

                # Optional: copy paper-ready PNGs into a stable directory.
                paper_dir_raw = getattr(self.config, "dump_attention_paper_dir", None)
                paper_tag = getattr(self.config, "dump_attention_paper_tag", None)
                if (
                    isinstance(paper_dir_raw, str)
                    and paper_dir_raw
                    and isinstance(paper_tag, str)
                    and paper_tag
                    and str(tag) == "student"
                ):
                    _copy_attention_paper_pngs(
                        src_dir=model_dir,
                        dst_dir=Path(paper_dir_raw),
                        case_id=str(cid),
                        tag=str(paper_tag),
                    )


def _assign_attention_viz_ids(model: nn.Module) -> None:
    """Ensure attention layers have stable viz ids/names for ctx recording."""
    try:
        i = 0
        for name, mod in model.named_modules():
            if isinstance(mod, AttentionLayer):
                try:
                    mod._viz_index = int(i)  # type: ignore[attr-defined]
                    mod._viz_name = str(name)  # type: ignore[attr-defined]
                except Exception:
                    pass
                i += 1
    except Exception:
        return


def _find_anchor_token_index(tokens: list[str], *, anchor: str) -> int | None:
    """Best-effort: find the first token index that contains `anchor`."""
    if not anchor:
        return None
    for i, t in enumerate(tokens):
        if anchor in t:
            return int(i)
    # Fallback: try the reconstructed string.
    # Use itertools.accumulate to avoid O(n²) string concatenation.
    from itertools import accumulate
    for i, s in enumerate(accumulate(tokens)):
        if anchor in s:
            return int(i)
    return None


def _attention_mass_summary(event: dict[str, Any], *, split: int, tokens: list[str]) -> dict[str, Any]:
    """Compute exemplar-vs-target attention mass for the final query token, per layer."""
    out: dict[str, Any] = {"split": int(split), "anchor_tokens": tokens[max(0, split - 2) : min(len(tokens), split + 3)]}
    layers = event.get("layers", [])
    summaries: list[dict[str, Any]] = []
    if not isinstance(layers, list):
        out["layers"] = summaries
        return out

    for layer in layers:
        try:
            idx = int(layer.get("index", -1))
            name = str(layer.get("name", ""))
            mode = str(layer.get("mode", ""))
            attn = layer.get("attn", None)
            if not isinstance(attn, dict):
                continue
            mats = attn.get("matrices", None)
            if not isinstance(mats, list) or not mats:
                continue
            # mats: list[head][tq][tk]
            # Mean over heads.
            tq = len(mats[0])
            tk = len(mats[0][0]) if tq > 0 else 0
            if tq <= 0 or tk <= 0:
                continue
            # Mean last-row distribution across heads.
            last = [0.0] * tk
            h_count = 0
            for h in mats:
                if not isinstance(h, list) or len(h) != tq:
                    continue
                row = h[-1]
                if not isinstance(row, list) or len(row) != tk:
                    continue
                for j in range(tk):
                    try:
                        last[j] += float(row[j])
                    except Exception:
                        pass
                h_count += 1
            if h_count > 0:
                last = [x / float(h_count) for x in last]

            split2 = max(0, min(int(split), tk))
            mass_exemplar = float(sum(last[:split2]))
            mass_target = float(sum(last[split2:tk]))
            # Top-k tokens by attention weight (from the mean row).
            topk = 12
            pairs = sorted([(float(w), int(i)) for i, w in enumerate(last)], reverse=True)[:topk]
            tops = [{"i": i, "w": w, "tok": tokens[i] if i < len(tokens) else ""} for (w, i) in pairs]

            summaries.append(
                {
                    "index": idx,
                    "name": name,
                    "mode": mode,
                    "tq": int(tq),
                    "tk": int(tk),
                    "mass_exemplar": mass_exemplar,
                    "mass_target": mass_target,
                    "top": tops,
                }
            )
        except Exception:
            continue

    out["layers"] = summaries
    return out


def _render_attention_pngs(
    *,
    model_dir: Path,
    event: dict[str, Any],
    tokens: list[str],
    split: int,
    case_id: str,
    model_tag: str,
) -> None:
    """Render two complementary PNG styles from the dumped attention JSON.

    Style 1 (best for diagnosis): layer × key-position heatmap of the final query token.
    Style 2 (best for presentation): last-layer per-head attention matrices (small grid).
    """
    if plt is None or np is None:
        return
    layers = event.get("layers", [])
    if not isinstance(layers, list) or not layers:
        return

    # Collect per-layer last-row (mean over heads).
    rows: list[Any] = []
    names: list[str] = []
    last_layer_heads: list[Any] | None = None

    for layer in layers:
        attn = layer.get("attn", None)
        if not isinstance(attn, dict):
            continue
        mats = attn.get("matrices", None)
        if not isinstance(mats, list) or not mats:
            continue
        # mats: list[head][tq][tk]
        try:
            head_arrays: list[Any] = []
            for h in mats:
                a = np.asarray(h, dtype=np.float32)
                if a.ndim != 2:
                    continue
                head_arrays.append(a)
            if not head_arrays:
                continue
            m = np.stack(head_arrays, axis=0)  # type: ignore[arg-type]  # (H,tq,tk)
            last = m[:, -1, :].mean(axis=0)  # (tk,)
        except Exception:
            continue

        rows.append(last)
        names.append(str(layer.get("name", "")))
        last_layer_heads = head_arrays  # overwritten; ends as last valid layer

    if not rows:
        return

    M = np.stack(rows, axis=0)  # type: ignore[arg-type]  # (L,tk)
    L, tk = int(M.shape[0]), int(M.shape[1])
    split2 = max(0, min(int(split), tk))

    # ---- Style 1: layer × key heatmap (final query token) ----
    try:
        fig = plt.figure(figsize=(min(14.0, 0.18 * tk + 4.0), min(10.0, 0.28 * L + 3.0)))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.axvline(split2 - 0.5, color="w", linewidth=1.5, alpha=0.9)
        ax.set_title(f"{case_id} • {model_tag} • final-query attention by layer (mean over heads)")
        ax.set_xlabel("key position (prompt tokens)")
        ax.set_ylabel("layer (sampled attention modules)")
        # Keep ticks lightweight (big prompts).
        if tk <= 64:
            ax.set_xticks(list(range(tk)))
            ax.set_xticklabels([t if len(t) <= 6 else t[:6] + "…" for t in tokens[:tk]], rotation=90, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="attention weight")
        fig.tight_layout()
        fig.savefig(model_dir / "attn_style1_layer_by_token.png", dpi=200)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

    # ---- Style 1b: last-row line plot (exemplar vs target mass across layers) ----
    try:
        mass_ex = M[:, :split2].sum(axis=1)
        mass_tg = M[:, split2:].sum(axis=1)
        fig = plt.figure(figsize=(10.5, 4.0))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(mass_ex, label="mass on exemplar region", linewidth=2)
        ax.plot(mass_tg, label="mass on target region", linewidth=2)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("layer (sampled attention modules)")
        ax.set_ylabel("attention mass (final query token)")
        ax.set_title(f"{case_id} • {model_tag} • final-query attention mass vs depth")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(model_dir / "attn_style1b_mass_by_layer.png", dpi=200)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

    # ---- Style 2: last-layer head grid (tq × tk) ----
    if not last_layer_heads:
        return
    try:
        H = int(len(last_layer_heads))
        ncols = min(4, H)
        nrows = int((H + ncols - 1) // ncols)
        fig = plt.figure(figsize=(min(14.0, 3.2 * ncols), min(10.0, 2.8 * nrows)))
        for i, a in enumerate(last_layer_heads):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            im = ax.imshow(a, aspect="auto", interpolation="nearest", cmap="magma")
            ax.axvline(split2 - 0.5, color="w", linewidth=1.0, alpha=0.9)
            ax.set_title(f"head {i}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{case_id} • {model_tag} • last sampled layer heads (tq×tk)", y=1.02)
        fig.tight_layout()
        fig.savefig(model_dir / "attn_style2_last_layer_heads.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass


def _copy_attention_paper_pngs(*, src_dir: Path, dst_dir: Path, case_id: str, tag: str) -> None:
    """Copy key PNG artifacts into a stable paper figure directory."""
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        mapping = {
            "attn_style1b_mass_by_layer.png": f"{case_id}_{tag}_mass_by_layer.png",
            "attn_style1_layer_by_token.png": f"{case_id}_{tag}_heatmap.png",
            "attn_style2_last_layer_heads.png": f"{case_id}_{tag}_last_layer_heads.png",
        }
        for src_name, dst_name in mapping.items():
            src = src_dir / src_name
            if not src.exists():
                continue
            dst = dst_dir / dst_name
            shutil.copy2(src, dst)
    except Exception:
        return

