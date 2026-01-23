"""Run multi-model benchmarks in a dedicated subprocess.

This provides stronger isolation than in-process load/unload, especially on MPS,
where allocator fragmentation and caching can leak across model runs.

The parent process writes a job.json containing:
- checkpoint specs (name/checkpoint/model_config/is_baseline)
- benchmark suite (pydantic dump)
- device/dtype/strict/unsafe_pickle_load
- metadata + output_dir + baseline_name

This worker loads models on-demand (one at a time) and runs the same
MultiModelBenchmarkRunner pipeline, then writes an artifacts index file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from benchmark.artifacts import ExperimentMetadata
from benchmark.multi_model_runner import MultiModelBenchmarkRunner
from carmath import weight_dtype
from config.benchmark import BenchmarkSuite, BenchmarkType
from console import logger
from data.tokenizers.base import Tokenizer
from data.tokenizers.builder import TokenizerBuilder
from infer.generate import GenerateConfig, Generator
from model import Model
from trainer.checkpoint_compare import _lower_and_validate_model_config, _safe_load_checkpoint


def _load_job(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _incompatible_keys_to_lists(obj: object) -> tuple[list[str], list[str]]:
    """Normalize torch load_state_dict return to (missing, unexpected) lists."""
    missing = getattr(obj, "missing_keys", None)
    unexpected = getattr(obj, "unexpected_keys", None)
    if isinstance(missing, list) and isinstance(unexpected, list):
        return [str(k) for k in missing], [str(k) for k in unexpected]
    # Torch also supports tuple-unpacking on the return in some versions.
    try:
        a, b = obj  # type: ignore[misc]
        if isinstance(a, list) and isinstance(b, list):
            return [str(k) for k in a], [str(k) for k in b]
    except Exception:
        pass
    return [], []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run multi-model benchmarks in an isolated subprocess.")
    parser.add_argument("--job", type=Path, required=True, help="Path to job JSON produced by parent.")
    args = parser.parse_args(argv)

    job = _load_job(args.job)
    output_dir = Path(str(job["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct suite + metadata
    suite = BenchmarkSuite(**job["suite"])
    meta_raw = job.get("metadata", {})
    metadata = ExperimentMetadata(**meta_raw) if isinstance(meta_raw, dict) else ExperimentMetadata(
        name="experiment", timestamp="", manifest_path="", teacher_checkpoint="", student_config="", device="", notes=""
    )

    device = torch.device(str(job.get("device", "cpu")))
    dtype_str = str(job.get("dtype", "auto")).lower().strip()
    strict = bool(job.get("strict", True))
    unsafe_pickle_load = bool(job.get("unsafe_pickle_load", False))
    baseline_name = job.get("baseline_name")

    dt = weight_dtype(device, dtype_str if dtype_str != "auto" else "auto")

    # Try to build a tokenizer for lightweight generation diagnostics.
    # (Used only for the load-time "decode probe", not required for benchmarks.)
    tokenizer: Tokenizer | None = None
    try:
        tok_cfg = None
        for spec in suite.benchmarks:
            cfg = getattr(spec, "config", None)
            if getattr(cfg, "type", None) == BenchmarkType.BEHAVIOR_INSTRUCT:
                tok_cfg = getattr(cfg, "tokenizer", None)
                break
        if tok_cfg is not None:
            tokenizer = TokenizerBuilder().build(tok_cfg)
    except Exception as e:
        logger.warning(f"Tokenizer init for diagnostics failed (continuing): {e!r}")

    checkpoint_specs = job.get("checkpoint_specs", [])
    specs_by_name: dict[str, dict[str, Any]] = {}
    for s in checkpoint_specs:
        if isinstance(s, dict) and "name" in s:
            specs_by_name[str(s["name"])] = s

    model_names = sorted(specs_by_name.keys())
    if not model_names:
        raise ValueError("No checkpoint_specs provided.")

    logger.header("Isolated Benchmarks", f"{len(model_names)} models • device={device.type} dtype={dt}")

    def load_model(name: str):
        spec = specs_by_name[str(name)]
        ckpt_path = Path(str(spec["checkpoint"]))
        cfg = _lower_and_validate_model_config(spec["model_config"])
        m = Model(cfg).to(device=device, dtype=dt)
        sd = _safe_load_checkpoint(ckpt_path, unsafe_pickle_load=unsafe_pickle_load)
        res = m.load_state_dict(sd, strict=strict)
        missing, unexpected = _incompatible_keys_to_lists(res)

        # Always emit a load report in isolation mode. This is critical when strict=false,
        # because missing/unexpected keys can silently result in partially-uninitialized models.
        try:
            model_sd = m.state_dict()
            report = {
                "model_name": str(name),
                "checkpoint": str(ckpt_path),
                "strict": bool(strict),
                "dtype": str(dt),
                "device": str(device),
                "num_checkpoint_tensors": int(len(sd)),
                "num_model_tensors": int(len(model_sd)),
                "missing_keys_count": int(len(missing)),
                "unexpected_keys_count": int(len(unexpected)),
                # Keep reports small but actionable; include a sample of keys.
                "missing_keys_head": missing[:50],
                "unexpected_keys_head": unexpected[:50],
            }

            # Lightweight decode probe: run a single greedy step and report what token
            # the model picks first (helps debug "blank outputs" that are actually whitespace/EOS).
            if tokenizer is not None:
                tok: Tokenizer = tokenizer

                def _probe(prompt: str) -> dict[str, Any]:
                    prompt_ids = tok.encode(prompt)
                    max_seq_len = max(128, int(len(prompt_ids) + 16))

                    # Next-token distribution (before any generation loop).
                    input_ids0 = torch.tensor([prompt_ids], device=device)
                    with torch.no_grad():
                        logits = m(input_ids0)
                    last = logits[0, -1]
                    vals, idx = torch.topk(last, k=10)
                    topk = []
                    for v, i in zip(vals.tolist(), idx.tolist()):
                        tok_id = int(i)
                        try:
                            tok_txt = str(tok.decode([tok_id]))
                        except Exception:
                            tok_txt = ""
                        topk.append({"id": tok_id, "text": tok_txt, "logit": float(v)})

                    # Short greedy generation preview.
                    gen_cfg = GenerateConfig(
                        max_new_tokens=8,
                        temperature=0.0,
                        max_seq_len=max_seq_len,
                        eos_token_id=tok.eos_token_id,
                    )
                    g = Generator(m, config=gen_cfg, device=device)
                    with torch.no_grad():
                        out = g.generate(input_ids0)
                    completion_ids = out[0, len(prompt_ids) :].tolist()
                    first_id = int(completion_ids[0]) if completion_ids else None
                    first_text = str(tok.decode([first_id])) if first_id is not None else ""
                    decoded_preview = str(tok.decode(completion_ids[:32])) if completion_ids else ""

                    eos = tok.eos_token_id
                    return {
                        "prompt": prompt,
                        "prompt_tokens": int(len(prompt_ids)),
                        "topk_next": topk,
                        "completion_token_ids": [int(x) for x in completion_ids[:32]],
                        "first_token_id": first_id,
                        "first_token_text": first_text,
                        "decoded_preview": decoded_preview,
                        "eos_token_id": eos,
                        "first_is_eos": bool(first_id is not None and eos is not None and first_id == int(eos)),
                    }

                report["decode_probes"] = [
                    _probe('User: Echo "HELLO".\n\nAssistant: '),   # current benchmark format
                    _probe('User: Echo "HELLO".\n\nAssistant:'),    # no trailing space
                    _probe('Echo "HELLO".\n\nAnswer:'),             # legacy v2 anchor
                ]

            report_path = output_dir / f"load_report_{str(name)}.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if missing or unexpected:
                raise ValueError(f"{name}: load_state_dict mismatch (missing={len(missing)}, unexpected={len(unexpected)}). See {report_path.name}")
        except Exception as e:
            logger.error(f"{name}: failed to write load report (continuing): {e!r}")
            raise e

        m.eval()
        return m

    def unload_model(m):
        try:
            m.to("cpu")
        except Exception:
            pass
        try:
            del m
        except Exception:
            pass
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    runner = MultiModelBenchmarkRunner(suite, device, metadata, baseline_name)
    paths = runner.run_isolated(model_names, load_model, unload_model)

    # Write index for parent to consume.
    index_path = output_dir / "artifacts_index.json"
    index = {k: str(v) for k, v in paths.items()}
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    logger.path(str(index_path), "artifacts_index")

    # Also write a minimal run manifest for debugging.
    job_out = output_dir / "isolated_job_echo.json"
    job_out.write_text(json.dumps(job, indent=2), encoding="utf-8")
    logger.path(str(job_out), "isolated_job_echo")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

