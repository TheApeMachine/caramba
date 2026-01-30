"""Helpers for loading benchmark models from `research/dba/*.yml` manifests.

This is meant for lightweight scripting (e.g. `scripts/bench.py`) where you want
to reuse manifest-defined checkpoints and model configs without running the full
experiment pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from carmath import weight_dtype
from compiler import Compiler
from config.manifest import Manifest
from config.model import ModelConfig
from config.target import ExperimentTargetConfig
from console import logger
from model import Model
from model.prompt_adapter import PromptTuningAdapter, load_prompt_embeddings
from trainer.checkpoint_compare import _lower_and_validate_model_config, _safe_load_checkpoint

from adapter.model import CompatibleWrapper, HFConfigShim

try:
    from peft import PeftModel  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    PeftModel = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BenchmarkCheckpointSpec:
    """A single model entry from a multi-checkpoint benchmark manifest."""

    name: str
    checkpoint: Path
    model_config: dict[str, Any]
    base_checkpoint: Path | None = None
    adapter_checkpoint: Path | None = None
    is_baseline: bool = False


def _normalize_dtype_spec(spec: str | None) -> str:
    """Normalize common dtype shorthands (fp16/fp32/bf16) for `weight_dtype()`."""
    if spec is None:
        return "auto"
    s = str(spec).strip().lower()
    return {
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
        "f32": "float32",
        "auto": "auto",
    }.get(s, s)


def _extract_model_dims(cfg: ModelConfig) -> dict[str, int]:
    """Infer minimal HF-like dims for PEFT compatibility shims."""
    from config.embedder import TokenEmbedderConfig
    from config.layer import AttentionLayerConfig
    from config.topology import (
        BranchingTopologyConfig,
        CyclicTopologyConfig,
        GraphTopologyConfig,
        NestedTopologyConfig,
        ParallelTopologyConfig,
        RecurrentTopologyConfig,
        ResidualTopologyConfig,
        SequentialTopologyConfig,
        StackedTopologyConfig,
    )

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
    num_attention_heads = int(attn_heads) if attn_heads is not None else max(1, int(hidden_size) // 64)

    return {
        "hidden_size": int(hidden_size),
        "num_attention_heads": int(num_attention_heads),
        "num_hidden_layers": int(num_hidden_layers),
        "vocab_size": int(vocab_size),
    }


def _resolve_entrypoint(manifest: Manifest, target: str) -> str:
    if manifest.entrypoints and target in manifest.entrypoints:
        target = manifest.entrypoints[target]
    if ":" in target:
        _, target = target.split(":", 1)
        target = target.strip()
    return target


def _find_experiment_target(manifest: Manifest, target: str) -> ExperimentTargetConfig:
    tname = _resolve_entrypoint(manifest, target)
    match = next((t for t in manifest.targets if getattr(t, "name", None) == tname), None)
    if match is None:
        raise ValueError(f"Target not found in manifest: {tname}")
    if not isinstance(match, ExperimentTargetConfig):
        raise ValueError(f"Target '{tname}' is not an experiment target.")
    return match


def read_multi_checkpoint_specs(
    *,
    manifest_path: str | Path,
    target: str = "multi_checkpoint_compare",
) -> tuple[list[BenchmarkCheckpointSpec], dict[str, Any]]:
    """Parse a multi-checkpoint compare target into loadable model specs.

    Returns:
      - specs: list of per-model checkpoint specs
      - meta: dict with keys like device/dtype/strict/unsafe_pickle_load when present
    """
    manifest_path = Path(manifest_path)
    manifest = Manifest.from_path(manifest_path)
    manifest = Compiler().compile(manifest)
    t = _find_experiment_target(manifest, target)

    cfg = dict(t.trainer.config or {})
    ckpts = cfg.get("checkpoints")
    if not isinstance(ckpts, list) or not ckpts:
        raise ValueError("Expected trainer.config.checkpoints to be a non-empty list.")

    specs: list[BenchmarkCheckpointSpec] = []
    for raw in ckpts:
        if not isinstance(raw, dict):
            raise ValueError("Each checkpoint spec must be a dict.")
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError("Checkpoint spec missing non-empty 'name'.")
        ckpt = raw.get("checkpoint")
        mc = raw.get("model_config")
        if not isinstance(ckpt, str) or not ckpt.strip():
            raise ValueError(f"{name}: checkpoint must be a non-empty string path.")
        if not isinstance(mc, dict):
            raise ValueError(f"{name}: model_config must be a dict.")

        base_ckpt = raw.get("base_checkpoint")
        adapter_ckpt = raw.get("adapter_checkpoint")
        specs.append(
            BenchmarkCheckpointSpec(
                name=name,
                checkpoint=Path(ckpt),
                model_config=dict(mc),
                base_checkpoint=Path(base_ckpt) if isinstance(base_ckpt, str) and base_ckpt else None,
                adapter_checkpoint=Path(adapter_ckpt) if isinstance(adapter_ckpt, str) and adapter_ckpt else None,
                is_baseline=bool(raw.get("is_baseline", False)),
            )
        )

    meta = {
        "device": cfg.get("device"),
        "dtype": cfg.get("dtype"),
        "strict": cfg.get("strict"),
        "unsafe_pickle_load": cfg.get("unsafe_pickle_load"),
    }
    return specs, meta


def _apply_adapter(
    *,
    base_model: Model,
    cfg: ModelConfig,
    adapter_path: Path,
    device: torch.device,
    dt: torch.dtype,
) -> nn.Module:
    adapter_dir = adapter_path.parent if adapter_path.is_file() else adapter_path
    cfg_path = adapter_dir / "adapter_config.json"

    peft_type = None
    if cfg_path.exists():
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
            peft_type = str(payload.get("peft_type", "")).lower()
        except Exception:
            peft_type = None

    if peft_type == "lora":
        if PeftModel is None:
            raise RuntimeError(
                "peft is required to load LoRA adapters. Install it (e.g. `uv add peft`) and retry."
            )
        dims = _extract_model_dims(cfg)
        hf_config = HFConfigShim(
            hidden_size=dims["hidden_size"],
            num_attention_heads=dims["num_attention_heads"],
            num_hidden_layers=dims["num_hidden_layers"],
            vocab_size=dims["vocab_size"],
        )
        wrapped = CompatibleWrapper(base_model, hf_config)
        model = PeftModel.from_pretrained(wrapped, str(adapter_dir))  # type: ignore[arg-type]
        return model.to(device=device, dtype=dt)

    # Default to prompt-tuning adapter semantics.
    prompt = load_prompt_embeddings(adapter_path)
    if prompt.ndim != 2:
        raise ValueError(f"prompt_embeddings must be rank-2, got {prompt.shape}")
    model = PromptTuningAdapter(base_model, prompt)
    return model.to(device=device, dtype=dt)


def load_models_from_benchmark_manifest(
    *,
    manifest_path: str | Path = "research/dba/benchmark-gated.yml",
    target: str = "multi_checkpoint_compare",
    models: Iterable[str] | None = None,
    device: str | torch.device | None = None,
    dtype: str | None = None,
    with_adapter: bool = False,
    strict: bool | None = None,
    unsafe_pickle_load: bool | None = None,
    adapter_overrides: dict[str, str | Path] | None = None,
) -> dict[str, nn.Module]:
    """Load benchmark models declared in a multi-checkpoint compare manifest.

    Args:
      manifest_path: Path to a manifest like `research/dba/benchmark-gated.yml`.
      target: Target name or entrypoint (default: "multi_checkpoint_compare").
      models: Optional subset of model names to load (e.g. ["baseline"]).
      device: Optional override ("mps", "cuda", "cpu"). Defaults to manifest.
      dtype: Optional override ("auto", "fp16", "bf16", "fp32"). Defaults to manifest.
      with_adapter: If True, load adapter checkpoints (LoRA or prompt-tuning) when present.
      strict: Optional override for `load_state_dict(strict=...)`. Defaults to manifest.
      unsafe_pickle_load: Optional override for unsafe checkpoint loading. Defaults to manifest.
      adapter_overrides: Optional {model_name: adapter_path_or_dir} overrides.

    Returns:
      Dict mapping model name -> loaded module (eval mode).
    """
    specs, meta = read_multi_checkpoint_specs(manifest_path=manifest_path, target=target)

    wanted = set(str(x) for x in models) if models is not None else None
    adapter_overrides = dict(adapter_overrides or {})

    # Resolve runtime defaults from the manifest unless overridden.
    dev = device if device is not None else meta.get("device")
    dev = torch.device(str(dev)) if not isinstance(dev, torch.device) else dev
    dt_spec = _normalize_dtype_spec(dtype if dtype is not None else meta.get("dtype"))
    dt = weight_dtype(dev, dt_spec)
    strict_load = bool(meta.get("strict")) if strict is None else bool(strict)
    unsafe = bool(meta.get("unsafe_pickle_load")) if unsafe_pickle_load is None else bool(unsafe_pickle_load)

    out: dict[str, nn.Module] = {}
    for spec in specs:
        if wanted is not None and spec.name not in wanted:
            continue

        ckpt_path = spec.base_checkpoint or spec.checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{spec.name}: checkpoint not found: {ckpt_path}")

        logger.subheader(f"Loading: {spec.name}")
        cfg = _lower_and_validate_model_config(dict(spec.model_config))
        model = Model(cfg).to(device=dev, dtype=dt)

        logger.info(f"  Checkpoint: {ckpt_path}")
        sd = _safe_load_checkpoint(ckpt_path, unsafe_pickle_load=unsafe)
        missing, unexpected = model.load_state_dict(sd, strict=strict_load)
        if missing or unexpected:
            logger.warning(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

        if with_adapter:
            adapter_path = adapter_overrides.get(spec.name, None)
            if adapter_path is not None:
                adapter_path = Path(adapter_path)
            else:
                adapter_path = spec.adapter_checkpoint
            if adapter_path is not None:
                if not adapter_path.exists():
                    raise FileNotFoundError(f"{spec.name}: adapter not found: {adapter_path}")
                logger.info(f"  Adapter: {adapter_path}")
                model = _apply_adapter(base_model=model, cfg=cfg, adapter_path=adapter_path, device=dev, dt=dt)

        model.eval()
        out[spec.name] = model

    if not out:
        available = ", ".join(s.name for s in specs)
        raise ValueError(f"No models loaded. Available: {available}")

    return out

