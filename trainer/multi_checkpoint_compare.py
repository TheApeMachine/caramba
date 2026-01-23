"""Multi-checkpoint comparison trainer (evaluation-only).

Loads N checkpoints with potentially different model configs, then returns them
to the engine so the multi-model benchmark suite can run and write artifacts.

This is meant for paper workflows where you have multiple checkpoints
(e.g., baseline vs DBA-sem16 vs DBA-sem8) and want a manifest-driven,
reproducible N-way comparison run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from carmath import weight_dtype
from console import logger
from model import Model
from trainer.checkpoint_compare import (
    _lower_and_validate_model_config,
    _safe_load_checkpoint,
)


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

            # Load weights
            logger.info(f"  Loading: {ckpt_path}")
            state_dict = _safe_load_checkpoint(
                ckpt_path, unsafe_pickle_load=self.unsafe_pickle_load
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
