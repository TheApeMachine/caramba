"""MLX execution engine for Apple Silicon.

This engine provides MLX-native execution for experiments, optimized for
Apple Silicon GPUs. Currently supports the routing hypothesis experiment
with attention surgery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from caramba.config.manifest import Manifest
from caramba.config.target import ExperimentTargetConfig
from caramba.console import logger


@dataclass(frozen=True, slots=True)
class MLXEngineContext:
    """Shared engine context passed to components at runtime."""

    backend: str = "mlx"


class MLXEngine:
    """MLX execution engine for Apple Silicon.

    This engine handles MLX-native model training, optimized for the
    unified memory architecture of Apple Silicon.
    """

    def __init__(self) -> None:
        # Verify MLX is available
        try:
            import mlx.core as mx
            self._mx = mx
            logger.info(f"MLX backend initialized (version: {getattr(mx, '__version__', 'unknown')})")
        except ImportError as e:
            raise ImportError(
                "MLX not available. Install with: pip install mlx"
            ) from e

    def run_experiment(
        self,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        *,
        dry_run: bool = False,
    ) -> Any:
        """Run an experiment using MLX backend.

        Currently supports:
        - trainer.gradient_isolation: Fresh DBA attention training (routing hypothesis)
        - trainer.attention_distillation: DBA with attention pattern distillation from teacher
        """
        trainer_ref = target.trainer.ref

        if trainer_ref == "trainer.gradient_isolation":
            # Check if distillation is enabled
            runs = target.runs
            if runs and runs[0].train:
                train_cfg = runs[0].train
                use_distill = getattr(train_cfg, "attention_distillation", False)
                if use_distill:
                    return self._run_attention_distillation(manifest, target, dry_run=dry_run)
            # Default: pure routing hypothesis (LM loss only)
            return self._run_routing_hypothesis(manifest, target, dry_run=dry_run)
        elif trainer_ref == "trainer.attention_distillation":
            return self._run_attention_distillation(manifest, target, dry_run=dry_run)
        elif trainer_ref == "trainer.dual_attention":
            return self._run_dual_attention(manifest, target, dry_run=dry_run)
        else:
            raise ValueError(
                f"MLX backend does not yet support trainer ref: {trainer_ref!r}. "
                f"Supported trainers: trainer.gradient_isolation, trainer.attention_distillation, trainer.dual_attention"
            )

    def _run_routing_hypothesis(
        self,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run the routing hypothesis experiment with MLX."""
        from caramba.trainer.mlx.routing_hypothesis import (
            run_routing_hypothesis_mlx,
            TrainConfig,
        )

        # Extract configuration from target
        runs = target.runs
        if not runs:
            raise ValueError("No runs defined in target")

        run = runs[0]  # Use first run
        train_cfg = run.train
        if train_cfg is None:
            raise ValueError("No train config in run")

        # Get data path
        data_path = target.data.config.get("path")
        if not data_path:
            raise ValueError("No data path specified in target.data.config.path")

        # Get teacher checkpoint
        teacher_ckpt = getattr(train_cfg, "teacher_ckpt", None)
        if not teacher_ckpt:
            raise ValueError("No teacher_ckpt specified in train config")

        # Resolve HuggingFace checkpoint path
        weights_path = self._resolve_weights_path(str(teacher_ckpt))

        # Get DBA dimensions from model config
        model_cfg = target.system.config.get("model", {})
        topology = model_cfg.get("topology", {})
        layers = topology.get("layers", [])

        # Find attention layer config (nested in topology)
        sem_dim = 256  # defaults
        geo_dim = 512
        v_dim = 768

        for layer in layers:
            if layer.get("type") == "NestedTopology":
                for sublayer in layer.get("layers", []):
                    if sublayer.get("type") == "ResidualTopology":
                        for inner in sublayer.get("layers", []):
                            if inner.get("type") == "AttentionLayer":
                                sem_dim = inner.get("sem_dim", sem_dim)
                                geo_dim = inner.get("geo_dim", geo_dim)
                                v_dim = inner.get("attn_dim", v_dim)
                                break

        # Training parameters
        max_steps = run.steps or 5000
        lr = getattr(train_cfg, "lr", 1e-4)
        batch_size = getattr(train_cfg, "batch_size", 1)
        block_size = getattr(train_cfg, "block_size", 2048)
        grad_accum = getattr(train_cfg, "gradient_accumulation_steps", 16)
        warmup_steps = getattr(train_cfg, "warmup_steps", 500)

        if dry_run:
            logger.info("Dry run - would execute MLX routing hypothesis training")
            logger.key_value({
                "weights_path": weights_path,
                "data_path": data_path,
                "sem_dim": sem_dim,
                "geo_dim": geo_dim,
                "v_dim": v_dim,
                "max_steps": max_steps,
                "lr": lr,
            })
            return {}

        # Run the experiment
        run_routing_hypothesis_mlx(
            teacher_weights_path=weights_path,
            data_path=data_path,
            sem_dim=sem_dim,
            geo_dim=geo_dim,
            v_dim=v_dim,
            max_steps=max_steps,
            lr=lr,
            batch_size=batch_size,
            block_size=block_size,
            grad_accum_steps=grad_accum,
            warmup_steps=warmup_steps,
        )

        return {"backend": "mlx", "status": "complete"}

    def _resolve_weights_path(self, checkpoint: str) -> str:
        """Resolve a checkpoint specification to an actual file path.

        Supports:
        - hf://org/model -> HuggingFace Hub download
        - Local file paths
        """
        if checkpoint.startswith("hf://"):
            model_id = checkpoint[5:]  # Remove "hf://" prefix
            return self._download_hf_weights(model_id)
        return checkpoint

    def _download_hf_weights(self, model_id: str) -> str:
        """Download weights from HuggingFace Hub and return local path."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError as e:
            raise ImportError(
                "huggingface_hub required for HF downloads. "
                "Install with: pip install huggingface-hub"
            ) from e

        logger.info(f"Resolving HuggingFace model: {model_id}")

        # Find the safetensors file
        try:
            files = list_repo_files(model_id)
            safetensor_files = [f for f in files if f.endswith(".safetensors")]

            if not safetensor_files:
                raise ValueError(f"No .safetensors files found in {model_id}")

            # Prefer model.safetensors, otherwise use first one
            target_file = "model.safetensors"
            if target_file not in safetensor_files:
                target_file = safetensor_files[0]

            local_path = hf_hub_download(model_id, target_file)
            logger.success(f"Downloaded weights to: {local_path}")
            return local_path

        except Exception as e:
            raise RuntimeError(f"Failed to download {model_id}: {e}") from e

    def _run_attention_distillation(
        self,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run attention distillation experiment with MLX.

        This trains DBA attention to mimic the teacher's attention patterns,
        giving the student a "hint" about where to look. Much faster convergence
        than pure LM loss.
        """
        from caramba.trainer.mlx.attention_distillation import run_attention_distillation

        # Extract configuration from target
        runs = target.runs
        if not runs:
            raise ValueError("No runs defined in target")

        run = runs[0]
        train_cfg = run.train
        if train_cfg is None:
            raise ValueError("No train config in run")

        # Get data path
        data_path = target.data.config.get("path")
        if not data_path:
            raise ValueError("No data path specified in target.data.config.path")

        # Get teacher checkpoint
        teacher_ckpt = getattr(train_cfg, "teacher_ckpt", None)
        if not teacher_ckpt:
            raise ValueError("No teacher_ckpt specified in train config")

        weights_path = self._resolve_weights_path(str(teacher_ckpt))

        # Get DBA dimensions from model config
        model_cfg = target.system.config.get("model", {})
        topology = model_cfg.get("topology", {})
        layers = topology.get("layers", [])

        sem_dim = 256
        geo_dim = 512
        v_dim = 768

        for layer in layers:
            if layer.get("type") == "NestedTopology":
                for sublayer in layer.get("layers", []):
                    if sublayer.get("type") == "ResidualTopology":
                        for inner in sublayer.get("layers", []):
                            if inner.get("type") == "AttentionLayer":
                                sem_dim = inner.get("sem_dim", sem_dim)
                                geo_dim = inner.get("geo_dim", geo_dim)
                                v_dim = inner.get("attn_dim", v_dim)
                                break

        # Training parameters
        max_steps = run.steps or 5000
        lr = getattr(train_cfg, "lr", 1e-4)
        batch_size = getattr(train_cfg, "batch_size", 1)
        block_size = getattr(train_cfg, "block_size", 2048)
        warmup_steps = getattr(train_cfg, "warmup_steps", 500)

        # Distillation-specific parameters
        distill_alpha = getattr(train_cfg, "distill_alpha", 1.0)
        lm_alpha = getattr(train_cfg, "lm_alpha", 0.1)

        if dry_run:
            logger.info("Dry run - would execute MLX attention distillation training")
            logger.key_value({
                "weights_path": weights_path,
                "data_path": data_path,
                "sem_dim": sem_dim,
                "geo_dim": geo_dim,
                "v_dim": v_dim,
                "max_steps": max_steps,
                "lr": lr,
                "distill_alpha": distill_alpha,
                "lm_alpha": lm_alpha,
            })
            return {}

        # Run the experiment
        run_attention_distillation(
            teacher_weights_path=weights_path,
            data_path=data_path,
            sem_dim=sem_dim,
            geo_dim=geo_dim,
            v_dim=v_dim,
            max_steps=max_steps,
            lr=lr,
            distill_alpha=distill_alpha,
            lm_alpha=lm_alpha,
            batch_size=batch_size,
            block_size=block_size,
            warmup_steps=warmup_steps,
        )

        return {"backend": "mlx", "status": "complete", "mode": "attention_distillation"}

    def _run_dual_attention(
        self,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run dual attention training experiment with MLX.

        This runs both original and DBA attention in parallel, training DBA
        to match the original attention output. This learns the optimal
        semantic/geometric decomposition.
        """
        from caramba.trainer.mlx.dual_attention import run_dual_attention_training

        # Extract configuration from target
        runs = target.runs
        if not runs:
            raise ValueError("No runs defined in target")

        run = runs[0]
        train_cfg = run.train
        if train_cfg is None:
            raise ValueError("No train config in run")

        # Get data path
        data_path = target.data.config.get("path")
        if not data_path:
            raise ValueError("No data path specified in target.data.config.path")

        # Get teacher checkpoint
        teacher_ckpt = getattr(train_cfg, "teacher_ckpt", None)
        if not teacher_ckpt:
            raise ValueError("No teacher_ckpt specified in train config")

        weights_path = self._resolve_weights_path(str(teacher_ckpt))

        # Get DBA dimensions from model config
        model_cfg = target.system.config.get("model", {})
        topology = model_cfg.get("topology", {})
        layers = topology.get("layers", [])

        sem_dim = 256
        geo_dim = 512
        v_dim = 768

        for layer in layers:
            if layer.get("type") == "NestedTopology":
                for sublayer in layer.get("layers", []):
                    if sublayer.get("type") == "ResidualTopology":
                        for inner in sublayer.get("layers", []):
                            if inner.get("type") == "AttentionLayer":
                                sem_dim = inner.get("sem_dim", sem_dim)
                                geo_dim = inner.get("geo_dim", geo_dim)
                                v_dim = inner.get("attn_dim", v_dim)
                                break

        # Training parameters
        max_steps = run.steps or 5000
        lr = getattr(train_cfg, "lr", 1e-4)
        batch_size = getattr(train_cfg, "batch_size", 1)
        block_size = getattr(train_cfg, "block_size", 2048)
        warmup_steps = getattr(train_cfg, "warmup_steps", 500)

        # Dual attention specific parameters
        output_match_alpha = getattr(train_cfg, "output_match_alpha", 1.0)
        lm_alpha = getattr(train_cfg, "lm_alpha", 0.01)

        if dry_run:
            logger.info("Dry run - would execute MLX dual attention training")
            logger.key_value({
                "weights_path": weights_path,
                "data_path": data_path,
                "sem_dim": sem_dim,
                "geo_dim": geo_dim,
                "v_dim": v_dim,
                "max_steps": max_steps,
                "lr": lr,
                "output_match_alpha": output_match_alpha,
                "lm_alpha": lm_alpha,
            })
            return {}

        # Run the experiment
        run_dual_attention_training(
            teacher_weights_path=weights_path,
            data_path=data_path,
            sem_dim=sem_dim,
            geo_dim=geo_dim,
            v_dim=v_dim,
            max_steps=max_steps,
            lr=lr,
            output_match_alpha=output_match_alpha,
            lm_alpha=lm_alpha,
            batch_size=batch_size,
            block_size=block_size,
            warmup_steps=warmup_steps,
        )

        return {"backend": "mlx", "status": "complete", "mode": "dual_attention"}
