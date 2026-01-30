"""Multi-checkpoint comparison trainer (evaluation-only).

Loads N checkpoints with potentially different model configs, then returns them
to the engine so the multi-model benchmark suite can run and write artifacts.

This is meant for paper workflows where you have multiple checkpoints
(e.g., baseline vs DBA-sem16 vs DBA-sem8) and want a manifest-driven,
reproducible N-way comparison run.
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import torch
from torch import nn

from carmath import weight_dtype
from console import logger
from model import Model
from adapter.model import CompatibleWrapper, HFConfigShim
from model.prompt_adapter import PromptTuningAdapter, load_prompt_embeddings
from trainer.checkpoint_compare import (
    _lower_and_validate_model_config,
    _safe_load_checkpoint,
)
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


class MultiCheckpointCompareTrainer:
    """Load N checkpoints and return modules for multi-model benchmarks."""

    def __init__(
        self,
        *,
        checkpoints: list[dict[str, Any]],
        device: str = "cpu",
        dtype: str = "auto",
        strict: bool = True,
        unsafe_pickle_load: bool = False,
        isolate_models: bool = True,
        process_isolation: bool = True,
    ) -> None:
        """Initialize the multi-checkpoint trainer.

        Args:
            checkpoints: List of checkpoint specifications, each containing:
                - name: Display name for the model (e.g., "baseline", "sem16")
                - checkpoint: Path to the checkpoint file
                - model_config: Model configuration dict (required)
                - is_baseline: If True, mark as baseline for delta calculations
            device: Device to load models on (e.g., "cuda", "cpu")
            dtype: Data type ("auto", "float16", "bfloat16", "float32")
            strict: If True, require exact state_dict key match
            unsafe_pickle_load: If True, allow loading pickled checkpoints
        """
        self.checkpoint_specs = list(checkpoints)
        self.device = torch.device(str(device))
        self.dtype = str(dtype).lower().strip()
        self.strict = bool(strict)
        self.unsafe_pickle_load = bool(unsafe_pickle_load)
        # If True, do not keep multiple models resident simultaneously.
        # Instead, return checkpoint specs so the engine can benchmark with
        # per-model load/run/unload isolation (important on MPS / limited VRAM).
        self.isolate_models = bool(isolate_models)
        # If True, ask the engine to run benchmarks in a separate process
        # (strongest isolation; avoids allocator / cache interference).
        self.process_isolation = bool(process_isolation)

        # Validate specs
        if not self.checkpoint_specs:
            raise ValueError("At least one checkpoint must be specified")
        for i, spec in enumerate(self.checkpoint_specs):
            if not isinstance(spec, dict):
                raise ValueError(f"Checkpoint spec {i} must be a dict")
            if "name" not in spec:
                raise ValueError(f"Checkpoint spec {i} missing 'name'")
            if "checkpoint" not in spec:
                raise ValueError(f"Checkpoint spec {i} missing 'checkpoint'")
            if "model_config" not in spec:
                raise ValueError(f"Checkpoint spec {i} missing 'model_config'")

    def run(
        self,
        *,
        manifest: Any,
        target: Any,
        engine: Any,  # unused; parity with other trainers
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        """Load all checkpoints and return models dict.

        Returns:
            dict with:
                - "models": dict[str, nn.Module] mapping model names to modules
                - "baseline_name": str | None, name of baseline model
                - "device": torch.device
                - "checkpoint_dir": str
        """
        if dry_run:
            return {
                "models": {spec["name"]: None for spec in self.checkpoint_specs},
                "checkpoint_specs": list(self.checkpoint_specs),
                "baseline_name": next((s["name"] for s in self.checkpoint_specs if s.get("is_baseline")), None)
                or (self.checkpoint_specs[0]["name"] if self.checkpoint_specs else None),
                "device": self.device,
                "dtype": self.dtype,
                "strict": self.strict,
                "unsafe_pickle_load": self.unsafe_pickle_load,
                "process_isolation": self.process_isolation,
            }

        dt = weight_dtype(self.device, self.dtype if self.dtype != "auto" else "auto")
        logger.header(
            "Multi-Checkpoint Compare",
            f"{len(self.checkpoint_specs)} models • device={self.device.type} dtype={dt}",
        )

        # Determine baseline name from specs (even in isolation mode).
        baseline_name: str | None = None
        for spec in self.checkpoint_specs:
            if bool(spec.get("is_baseline", False)):
                baseline_name = str(spec["name"])
                break
        if baseline_name is None and self.checkpoint_specs:
            baseline_name = str(self.checkpoint_specs[0]["name"])

        # Isolation mode: don't load all models upfront.
        if self.isolate_models:
            logger.info("Isolation mode enabled: benchmarks will load/unload one model at a time.")
            return {
                "checkpoint_specs": list(self.checkpoint_specs),
                "baseline_name": baseline_name,
                "device": self.device,
                "dtype": self.dtype,
                "strict": self.strict,
                "unsafe_pickle_load": self.unsafe_pickle_load,
                "process_isolation": self.process_isolation,
                "checkpoint_dir": str(
                    Path(getattr(manifest, "artifacts_dir", "artifacts")) / "benchmarks"
                ),
            }

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

        def _resolve_base_checkpoint(spec: dict[str, Any], adapter_path: Path) -> Path | None:
            base_ckpt = spec.get("base_checkpoint")
            if isinstance(base_ckpt, str) and base_ckpt:
                return Path(base_ckpt)
            cfg_path = adapter_path.parent / "adapter_config.json"
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    base = cfg.get("base_model_name_or_path")
                    if isinstance(base, str) and base:
                        return Path(base)
                except Exception:
                    pass
            for name in ("model.safetensors", "pytorch_model.bin", "model.pt"):
                cand = adapter_path.parent / name
                if cand.exists():
                    return cand
            return None

        models: dict[str, nn.Module] = {}

        for spec in self.checkpoint_specs:
            name = spec["name"]
            ckpt_path = Path(spec["checkpoint"])
            model_config = spec["model_config"]
            is_baseline = spec.get("is_baseline", False)

            logger.subheader(f"Loading: {name}")

            # Validate checkpoint exists
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            # Build and validate model config
            logger.info(f"  Config: {model_config.get('type', 'unknown')}")
            cfg = _lower_and_validate_model_config(model_config)

            # Create model
            model = Model(cfg).to(device=self.device, dtype=dt)

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

            # Load weights
            logger.info(f"  Loading: {base_ckpt_path}")
            state_dict = _safe_load_checkpoint(
                base_ckpt_path, unsafe_pickle_load=self.unsafe_pickle_load
            )
            result = model.load_state_dict(state_dict, strict=bool(self.strict))
            missing, unexpected = result
            if missing or unexpected:
                logger.warning(
                    f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}"
                )
                # Log details to help diagnose key mismatches
                if missing:
                    logger.warning(f"  Missing keys (first 5): {missing[:5]}")
                if unexpected:
                    logger.warning(f"  Unexpected keys (first 5): {unexpected[:5]}")

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
                    wrapped = CompatibleWrapper(model, hf_config)
                    peft_model = PeftModel.from_pretrained(wrapped, str(adapter_dir))  # type: ignore[arg-type]
                    model = peft_model.to(device=self.device, dtype=dt)
                    logger.info("  Adapter: LoRA")
                else:
                    prompt = load_prompt_embeddings(adapter_path)
                    if prompt.ndim != 2:
                        raise ValueError(f"{name}: prompt_embeddings must be rank-2, got {prompt.shape}")
                    model = PromptTuningAdapter(model, prompt)
                    logger.info(
                        f"  Adapter: prompt-tuning ({int(prompt.shape[0])} virtual tokens)"
                    )

            model.eval()
            models[name] = model

            if is_baseline:
                baseline_name = name
                logger.info("  Marked as baseline")

            # Log model info
            n_params = sum(p.numel() for p in model.parameters())
            logger.metric(name, n_params / 1e6, "M params")

        # If no explicit baseline, use first model
        if baseline_name is None and models:
            baseline_name = str(self.checkpoint_specs[0]["name"])
            logger.info(f"Using '{baseline_name}' as baseline (first model)")

        logger.success(f"Loaded {len(models)} models")

        return {
            "models": models,
            "baseline_name": baseline_name,
            "device": self.device,
            "checkpoint_dir": str(
                Path(getattr(manifest, "artifacts_dir", "artifacts")) / "benchmarks"
            ),
        }
