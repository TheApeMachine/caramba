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
import json
from typing import Any

import torch
import inspect

from benchmark.artifacts import ExperimentMetadata
from benchmark.multi_model_runner import MultiModelBenchmarkRunner
from carmath import weight_dtype
from config.benchmark import BenchmarkSuite, BenchmarkType
from console import logger
from data.tokenizers.base import Tokenizer
from data.tokenizers.builder import TokenizerBuilder
from infer.generate import GenerateConfig, Generator
from model import Model
from adapter.model import CompatibleWrapper, HFConfigShim
from model.prompt_adapter import PromptTuningAdapter, load_prompt_embeddings
from trainer.checkpoint_compare import _lower_and_validate_model_config, _safe_load_checkpoint
from config.layer import AttentionLayerConfig
from config.embedder import TokenEmbedderConfig
from config.topology import (
    BranchingTopologyConfig,
    CyclicTopologyConfig,
    NestedTopologyConfig,
    ParallelTopologyConfig,
    RecurrentTopologyConfig,
    ResidualTopologyConfig,
    SequentialTopologyConfig,
    StackedTopologyConfig,
    GraphTopologyConfig,
)

try:
    from peft import PeftConfig, PeftModel  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    PeftConfig = None  # type: ignore[assignment]
    PeftModel = None  # type: ignore[assignment]


def _extract_logits(output: Any) -> torch.Tensor:
    """Best-effort extraction of logits tensor from various model wrappers."""
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("logits")
    if hasattr(output, "logits"):
        output = output.logits
    if output is None:
        raise ValueError("Model did not return logits.")
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected logits Tensor, got {type(output).__name__}")
    return output


def _supports_ctx(model: torch.nn.Module) -> bool:
    """Best-effort check whether model.forward accepts `ctx`."""
    try:
        sig = inspect.signature(model.forward)
    except Exception:
        return False
    return "ctx" in sig.parameters


@torch.no_grad()
def _generate_greedy_no_ctx(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> torch.Tensor:
    """Tiny greedy generation fallback that does not require KV-cache/ctx.

    This is used only for load-time diagnostics when a model wrapper (e.g. PEFT)
    does not consume InferContext caches.
    """
    out_ids = input_ids
    for _ in range(int(max_new_tokens)):
        logits = _extract_logits(model(out_ids))
        if logits.dim() == 3:
            next_logits = logits[:, -1, :]
        elif logits.dim() == 2:
            next_logits = logits
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
        next_tok = next_logits.argmax(dim=-1)
        out_ids = torch.cat([out_ids, next_tok[:, None]], dim=1)
        if eos_token_id is not None and bool((next_tok == int(eos_token_id)).all()):
            break
    return out_ids

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

    def _resolve_base_checkpoint(spec: dict[str, Any], adapter_path: Path) -> Path | None:
        # 1) Explicit base_checkpoint in manifest
        base_ckpt = spec.get("base_checkpoint")
        if isinstance(base_ckpt, str) and base_ckpt:
            return Path(base_ckpt)

        # 2) Try adapter_config.json base_model_name_or_path
        cfg_path = adapter_path.parent / "adapter_config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                base = cfg.get("base_model_name_or_path")
                if isinstance(base, str) and base:
                    return Path(base)
            except Exception:
                pass

        # 3) Try common filenames in the same directory
        for name in ("model.safetensors", "pytorch_model.bin", "model.pt"):
            cand = adapter_path.parent / name
            if cand.exists():
                return cand

        return None

    def _extract_model_dims(cfg) -> dict[str, int]:
        def walk(node: object) -> tuple[int, int | None, int | None]:
            if isinstance(node, AttentionLayerConfig):
                return 1, int(node.n_heads), int(node.d_model)
            if isinstance(
                node,
                (
                    NestedTopologyConfig,
                    StackedTopologyConfig,
                    ResidualTopologyConfig,
                    SequentialTopologyConfig,
                    ParallelTopologyConfig,
                    BranchingTopologyConfig,
                    CyclicTopologyConfig,
                    RecurrentTopologyConfig,
                ),
            ):
                total = 0
                heads: int | None = None
                d_model: int | None = None
                for layer in node.layers:
                    count, layer_heads, layer_d_model = walk(layer)
                    total += count
                    if heads is None and layer_heads is not None:
                        heads = layer_heads
                    if d_model is None and layer_d_model is not None:
                        d_model = layer_d_model
                total *= int(node.repeat)
                return total, heads, d_model
            if isinstance(node, GraphTopologyConfig):
                return 0, None, None
            return 0, None, None

        attn_layers, attn_heads, attn_d_model = walk(cfg.topology)
        hidden_size = None
        vocab_size = None
        if isinstance(cfg.embedder, TokenEmbedderConfig):
            hidden_size = int(cfg.embedder.d_model)
            vocab_size = int(cfg.embedder.vocab_size)
        if hidden_size is None:
            hidden_size = attn_d_model
        if vocab_size is None and cfg.vocab_size is not None:
            vocab_size = int(cfg.vocab_size)
        if hidden_size is None or vocab_size is None:
            raise ValueError("Could not infer hidden_size/vocab_size from model config")
        num_hidden_layers = int(attn_layers) if attn_layers > 0 else int(getattr(cfg.weight_init, "n_layers", 1))
        num_attention_heads = int(attn_heads) if attn_heads is not None else max(1, hidden_size // 64)
        return {
            "hidden_size": int(hidden_size),
            "num_attention_heads": int(num_attention_heads),
            "num_hidden_layers": int(num_hidden_layers),
            "vocab_size": int(vocab_size),
        }

    def load_model(name: str):
        spec = specs_by_name[str(name)]
        ckpt_path = Path(str(spec["checkpoint"]))
        cfg = _lower_and_validate_model_config(spec["model_config"])
        m = Model(cfg).to(device=device, dtype=dt)
        adapter_ckpt = spec.get("adapter_checkpoint")
        adapter_path = Path(str(adapter_ckpt)) if isinstance(adapter_ckpt, str) else None
        if adapter_path is None and ckpt_path.name == "adapter_model.safetensors":
            adapter_path = ckpt_path
        adapter_dir = adapter_path.parent if adapter_path is not None else None

        base_ckpt_path = ckpt_path
        if adapter_path is not None:
            resolved_base = _resolve_base_checkpoint(spec, adapter_path)
            if resolved_base is None:
                raise ValueError(
                    f"{name}: adapter checkpoint requires a base checkpoint. "
                    "Set 'base_checkpoint' in the manifest or ensure adapter_config.json includes "
                    "'base_model_name_or_path'."
                )
            base_ckpt_path = resolved_base

        sd = _safe_load_checkpoint(base_ckpt_path, unsafe_pickle_load=unsafe_pickle_load)
        res = m.load_state_dict(sd, strict=strict)
        missing, unexpected = _incompatible_keys_to_lists(res)

        if adapter_path is not None:
            peft_type = None
            if adapter_dir is not None and (adapter_dir / "adapter_config.json").exists():
                try:
                    cfg_payload = json.loads(
                        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
                    )
                    peft_type = str(cfg_payload.get("peft_type", "")).lower()
                except Exception:
                    peft_type = None

            if peft_type == "lora":
                if PeftConfig is None or PeftModel is None:
                    raise RuntimeError("peft is required to load LoRA adapters.")
                dims = _extract_model_dims(cfg)
                hf_config = HFConfigShim(
                    hidden_size=dims["hidden_size"],
                    num_attention_heads=dims["num_attention_heads"],
                    num_hidden_layers=dims["num_hidden_layers"],
                    vocab_size=dims["vocab_size"],
                )
                wrapped = CompatibleWrapper(m, hf_config)
                peft_model = PeftModel.from_pretrained(wrapped, str(adapter_dir))  # type: ignore[arg-type]
                m = peft_model.to(device=device, dtype=dt)
                logger.info(f"{name}: loaded LoRA adapter.")
            else:
                prompt = load_prompt_embeddings(adapter_path)
                if prompt.ndim != 2:
                    raise ValueError(f"{name}: prompt_embeddings must be rank-2, got {prompt.shape}")
                m = PromptTuningAdapter(m, prompt)
                logger.info(
                    f"{name}: loaded prompt-tuning adapter ({int(prompt.shape[0])} virtual tokens)."
                )

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
                        out0 = m(input_ids0)
                    logits0 = _extract_logits(out0)
                    # Support both (B, T, V) and (B, V) shapes.
                    if logits0.dim() == 3:
                        last = logits0[0, -1]
                    elif logits0.dim() == 2:
                        last = logits0[0]
                    else:
                        raise ValueError(f"Unexpected logits shape: {tuple(logits0.shape)}")
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
                    out: torch.Tensor
                    # Some wrappers (notably PEFT) do not pass `ctx` through, which causes
                    # InferContext.ensure_consumed() to fail. Fall back to a tiny no-ctx loop.
                    if _supports_ctx(m):
                        try:
                            g = Generator(m, config=gen_cfg, device=device)
                            out = g.generate(input_ids0)
                        except Exception:
                            out = _generate_greedy_no_ctx(
                                model=m,
                                input_ids=input_ids0,
                                max_new_tokens=int(gen_cfg.max_new_tokens),
                                eos_token_id=tok.eos_token_id,
                            )
                    else:
                        out = _generate_greedy_no_ctx(
                            model=m,
                            input_ids=input_ids0,
                            max_new_tokens=int(gen_cfg.max_new_tokens),
                            eos_token_id=tok.eos_token_id,
                        )
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

                try:
                    report["decode_probes"] = [
                        _probe('User: Echo "HELLO".\n\nAssistant: '),  # current benchmark format
                        _probe('User: Echo "HELLO".\n\nAssistant:'),  # no trailing space
                        _probe('Echo "HELLO".\n\nAnswer:'),  # legacy v2 anchor
                    ]
                except Exception as e:
                    # Diagnostics must not kill benchmark runs.
                    report["decode_probes_error"] = repr(e)
                    report["decode_probes"] = []

            report_path = output_dir / f"load_report_{str(name)}.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if (missing or unexpected) and strict:
                raise ValueError(
                    f"{name}: load_state_dict mismatch (missing={len(missing)}, unexpected={len(unexpected)}). "
                    f"See {report_path.name}"
                )
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

