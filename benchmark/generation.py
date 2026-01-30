"""Generation benchmark: qualitative outputs + degeneration probes.

This benchmark is intentionally lightweight and works for base LMs:
- encode a raw prompt with a configured tokenizer
- generate with Caramba's `infer.generate.Generator` (KV-cache aware)
- optionally apply a repetition-penalty "breakpoint" probe
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from config.benchmark import GenerationBenchmarkConfig
from console import logger
from data.tokenizers.builder import TokenizerBuilder
from infer.generate import GenerateConfig, Generator, sample_next_token, _apply_repetition_penalty_


@dataclass
class GenerationCase:
    id: str
    prompt: str


@dataclass
class GenerationCaseResult:
    case_id: str
    model_name: str
    prompt_chars: int
    completion: str
    completion_chars: int
    # Degeneration metrics (token-level where possible; string-level fallback).
    break_step: int | None = None
    unique_token_ratio: float | None = None
    dominant_token_id: int | None = None
    dominant_token_frac: float | None = None
    longest_run: int | None = None


@dataclass
class GenerationResult:
    model_name: str
    cases: list[GenerationCaseResult] = field(default_factory=list)


def _tokenizer_vocab_size(tok: object) -> int | None:
    """Best-effort tokenizer vocab size for masking padded logits."""
    if hasattr(tok, "n_vocab"):
        try:
            v = int(getattr(tok, "n_vocab"))
            return v if v > 0 else None
        except Exception:
            return None
    if hasattr(tok, "vocab_size"):
        try:
            v = int(getattr(tok, "vocab_size"))
            return v if v > 0 else None
        except Exception:
            return None
    return None


def _mask_invalid_vocab(logits: Tensor, *, valid_vocab_size: int | None) -> Tensor:
    """Mask logits beyond valid vocab (returns a safe tensor)."""
    if valid_vocab_size is None:
        return logits
    vv = int(valid_vocab_size)
    if vv <= 0 or int(logits.size(-1)) <= vv:
        return logits
    out = logits.clone()
    out[..., vv:] = float("-inf")
    return out


def _safe_name(s: str) -> str:
    t = str(s).strip() or "case"
    for ch in ("/", "\\", ":", " ", "\t", "\n"):
        t = t.replace(ch, "_")
    return t


def _load_cases(path: str | Path) -> list[GenerationCase]:
    p = Path(str(path))
    raw = p.read_text(encoding="utf-8")
    import yaml

    data = yaml.safe_load(raw)
    cases: list[GenerationCase] = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str):
                cases.append(GenerationCase(id=f"prompt_{i}", prompt=item))
            elif isinstance(item, dict):
                pid = str(item.get("id", f"prompt_{i}"))
                pr = str(item.get("prompt", ""))
                cases.append(GenerationCase(id=pid, prompt=pr))
    elif isinstance(data, dict):
        # Accept {"prompts": [...]}.
        ps = data.get("prompts")
        if isinstance(ps, list):
            for i, item in enumerate(ps):
                if isinstance(item, str):
                    cases.append(GenerationCase(id=f"prompt_{i}", prompt=item))
                elif isinstance(item, dict):
                    pid = str(item.get("id", f"prompt_{i}"))
                    pr = str(item.get("prompt", ""))
                    cases.append(GenerationCase(id=pid, prompt=pr))
    else:
        raise ValueError(f"Unsupported prompts_file format: {type(data).__name__}")
    # Filter empties.
    cases = [c for c in cases if str(c.prompt).strip()]
    if not cases:
        raise ValueError(f"No prompts found in {p}")
    return cases


def _degeneration_from_tokens(tokens: list[int], *, target_id: int | None) -> dict[str, Any]:
    if not tokens:
        return {
            "break_step": None,
            "unique_token_ratio": 0.0,
            "dominant_token_id": None,
            "dominant_token_frac": 0.0,
            "longest_run": 0,
        }
    # Break step: first token != target_id (only meaningful if target_id provided).
    break_step = None
    if target_id is not None:
        for i, t in enumerate(tokens):
            if int(t) != int(target_id):
                break_step = int(i)
                break
    # Dominant token.
    counts: dict[int, int] = {}
    for t in tokens:
        tid = int(t)
        counts[tid] = counts.get(tid, 0) + 1
    dom_id, dom_cnt = max(counts.items(), key=lambda kv: kv[1])
    uniq_ratio = float(len(counts)) / float(len(tokens)) if tokens else 0.0
    # Longest run of identical consecutive token IDs.
    longest = 1
    run = 1
    for i in range(1, len(tokens)):
        if int(tokens[i]) == int(tokens[i - 1]):
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    return {
        "break_step": break_step,
        "unique_token_ratio": float(uniq_ratio),
        "dominant_token_id": int(dom_id),
        "dominant_token_frac": float(dom_cnt / float(len(tokens))) if tokens else 0.0,
        "longest_run": int(longest),
    }


class GenerationBenchmark:
    def __init__(self, config: GenerationBenchmarkConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.tokenizer = TokenizerBuilder().build(config.tokenizer)
        self._cases: list[GenerationCase] | None = None

    def _cases_list(self) -> list[GenerationCase]:
        if self._cases is None:
            self._cases = _load_cases(self.config.prompts_file)
        return self._cases

    def run(self, model: nn.Module, model_name: str, *, output_dir: Path) -> GenerationResult:
        model.eval()
        cases = self._cases_list()

        # Resolve tokenizer vocab size (to avoid sampling padded tokens).
        valid_vocab_size = _tokenizer_vocab_size(self.tokenizer)

        # Resolve optional target token id (best-effort).
        target_id: int | None = None
        if self.config.target_text:
            ids = self.tokenizer.encode(str(self.config.target_text))
            if len(ids) == 1:
                target_id = int(ids[0])
            else:
                logger.warning(
                    f"generation: target_text tokenized to {len(ids)} tokens; breakpoint metrics will be disabled."
                )

        out_dir = Path(output_dir) / "generation" / _safe_name(str(model_name))
        out_dir.mkdir(parents=True, exist_ok=True)
        trace_path = out_dir / "repeat_penalty_trace.jsonl"

        # Tokenize stop strings (token-id suffix sequences).
        stop_sequences: list[list[int]] = []
        stops = getattr(self.config, "stop", None)
        if isinstance(stops, list):
            for s in stops:
                ss = str(s)
                if not ss:
                    continue
                try:
                    ids = list(self.tokenizer.encode(ss))
                except Exception:
                    ids = []
                if not ids:
                    continue
                stop_sequences.append([int(x) for x in ids])

        gen_cfg = GenerateConfig(
            max_new_tokens=int(self.config.max_new_tokens),
            temperature=float(self.config.temperature),
            top_p=float(self.config.top_p),
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            max_seq_len=2048,
            repetition_penalty=float(self.config.repetition_penalty),
            stop_sequences=stop_sequences,
        )
        g = Generator(model, config=gen_cfg, device=self.device)
        stop_sequences = list(getattr(gen_cfg, "stop_sequences", []) or [])

        results = GenerationResult(model_name=str(model_name))

        # Stream per-step traces (one JSON object per generated token).
        with open(trace_path, "w") as tf:
            for case in cases:
                prompt = str(case.prompt)
                prompt_ids = self.tokenizer.encode(prompt)
                input_ids = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)

                g.reset()
                logits = _mask_invalid_vocab(g.prefill(input_ids), valid_vocab_size=valid_vocab_size)
                generated_ids: list[int] = []

                # Manual decode loop so we can record per-step tokens and compute breakpoint.
                # This mirrors Generator.generate(), including repetition penalty behavior.
                for step in range(int(self.config.max_new_tokens)):
                    if float(self.config.repetition_penalty) != 1.0:
                        # Build token_ids seen so far (prompt + generated).
                        if generated_ids:
                            seen = torch.cat(
                                [input_ids, torch.tensor([generated_ids], device=self.device, dtype=torch.long)],
                                dim=1,
                            )
                        else:
                            seen = input_ids
                        logits = _apply_repetition_penalty_(logits, token_ids=seen, penalty=float(self.config.repetition_penalty))
                    logits = _mask_invalid_vocab(logits, valid_vocab_size=valid_vocab_size)

                    next_tok = sample_next_token(
                        logits,
                        temperature=float(self.config.temperature),
                        top_k=None,
                        top_p=float(self.config.top_p),
                    )
                    tid = int(next_tok.item())
                    generated_ids.append(tid)

                    # Trace record (keep small but debuggable).
                    try:
                        tok_txt = str(self.tokenizer.decode([tid]))
                    except Exception:
                        tok_txt = ""
                    tf.write(
                        json.dumps(
                            {
                                "case_id": str(case.id),
                                "step": int(step),
                                "token_id": int(tid),
                                "token_text": tok_txt,
                            }
                        )
                        + "\n"
                    )

                    # EOS stop.
                    eos = getattr(self.tokenizer, "eos_token_id", None)
                    if eos is not None and int(tid) == int(eos):
                        break

                    # Stop-sequence stop (token-id suffix match against prompt+generated).
                    if stop_sequences:
                        full = prompt_ids + generated_ids
                        hit = False
                        for seq in stop_sequences:
                            k = int(len(seq))
                            if k > 0 and len(full) >= k and full[-k:] == seq:
                                hit = True
                                break
                        if hit:
                            break

                    logits = g.decode_step(torch.tensor([tid], device=self.device, dtype=torch.long))
                    logits = _mask_invalid_vocab(logits, valid_vocab_size=valid_vocab_size)

                # Decode completion.
                eos = getattr(self.tokenizer, "eos_token_id", None)
                completion_ids = generated_ids
                if eos is not None and completion_ids and int(completion_ids[-1]) == int(eos):
                    completion_ids = completion_ids[:-1]
                if stop_sequences and completion_ids:
                    for seq in sorted(stop_sequences, key=lambda x: len(x), reverse=True):
                        k = len(seq)
                        if k > 0 and len(prompt_ids) + len(completion_ids) >= k:
                            # Check suffix on prompt+completion for correctness.
                            full = prompt_ids + completion_ids
                            if full[-k:] == seq:
                                completion_ids = completion_ids[:-k]
                                break
                completion = str(self.tokenizer.decode(completion_ids)) if completion_ids else ""

                degen = _degeneration_from_tokens(generated_ids, target_id=target_id)
                results.cases.append(
                    GenerationCaseResult(
                        case_id=str(case.id),
                        model_name=str(model_name),
                        prompt_chars=int(len(prompt)),
                        completion=str(completion),
                        completion_chars=int(len(completion)),
                        break_step=degen["break_step"],
                        unique_token_ratio=float(degen["unique_token_ratio"]),
                        dominant_token_id=degen["dominant_token_id"],
                        dominant_token_frac=float(degen["dominant_token_frac"]),
                        longest_run=int(degen["longest_run"]),
                    )
                )

        # Summary CSV.
        summary_path = out_dir / "repeat_penalty_summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "case_id",
                    "model",
                    "prompt_chars",
                    "completion_chars",
                    "break_step",
                    "unique_token_ratio",
                    "dominant_token_id",
                    "dominant_token_frac",
                    "longest_run",
                ]
            )
            for r in results.cases:
                w.writerow(
                    [
                        r.case_id,
                        r.model_name,
                        r.prompt_chars,
                        r.completion_chars,
                        "" if r.break_step is None else int(r.break_step),
                        "" if r.unique_token_ratio is None else float(r.unique_token_ratio),
                        "" if r.dominant_token_id is None else int(r.dominant_token_id),
                        "" if r.dominant_token_frac is None else float(r.dominant_token_frac),
                        "" if r.longest_run is None else int(r.longest_run),
                    ]
                )

        # Small index file for convenience.
        (out_dir / "generation_result.json").write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")
        logger.path(str(summary_path), f"generation:{model_name}:summary")
        logger.path(str(trace_path), f"generation:{model_name}:trace")

        return results

