"""PyTorch execution engine.

This is the first concrete engine. Other engines (JAX/sklearn/etc.) can be
added later without changing the manifest schema: only registry mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from config.manifest import Manifest
from config.target import ExperimentTargetConfig
from runtime.registry import ComponentRegistry
from benchmark.artifacts import ExperimentMetadata
from benchmark.runner import BenchmarkRunner
from config.benchmark import BenchmarkSuite
from console import logger
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class EngineContext:
    """Shared engine context passed to components at runtime."""

    backend: str = "torch"


class TorchEngine:
    def __init__(self, *, registry: ComponentRegistry | None = None) -> None:
        self.registry = registry or ComponentRegistry()
        self._register_builtin_components()

    def _register_builtin_components(self) -> None:
        # Trainers
        self.registry.register(
            backend="torch",
            ref="trainer.upcycle",
            python="trainer.upcycle:UpcycleTrainer",
        )
        self.registry.register(
            backend="torch",
            ref="trainer.standard",
            python="trainer.standard:StandardTrainer",
        )
        self.registry.register(
            backend="torch",
            ref="trainer.gradient_isolation",
            python="trainer.gradient_isolation:GradientIsolationTrainer",
        )

        # Datasets
        self.registry.register(
            backend="torch",
            ref="dataset.tokens",
            python="data.token_dataset:TokenDataset",
        )
        self.registry.register(
            backend="torch",
            ref="dataset.npy_supervised",
            python="data.npy_supervised:NpySupervisedDataset",
        )
        self.registry.register(
            backend="torch",
            ref="dataset.graph_npy",
            python="data.graph_npy:GraphNpyDataset",
        )
        self.registry.register(
            backend="torch",
            ref="dataset.diffusion_vector",
            python="data.diffusion_vector:DiffusionVectorDataset",
        )
        self.registry.register(
            backend="torch",
            ref="dataset.tensors",
            python="data.tensors:TensorFilesDataset",
        )

        # Systems
        self.registry.register(
            backend="torch",
            ref="system.language_model",
            python="model.language_model_system:LanguageModelSystem",
        )
        self.registry.register(
            backend="torch",
            ref="system.generic",
            python="model.generic_system:GenericSystem",
        )
        self.registry.register(
            backend="torch",
            ref="system.mlp_classifier",
            python="model.mlp_classifier_system:MLPClassifierSystem",
        )
        self.registry.register(
            backend="torch",
            ref="system.gcn",
            python="model.gcn_system:GCNSystem",
        )
        self.registry.register(
            backend="torch",
            ref="system.diffusion_denoiser",
            python="model.diffusion_denoiser_system:DiffusionDenoiserSystem",
        )
        self.registry.register(
            backend="torch",
            ref="system.graph",
            python="model.graph_system:GraphSystem",
        )

        # Objectives
        self.registry.register(
            backend="torch",
            ref="objective.next_token_ce",
            python="trainer.objectives:NextTokenCrossEntropyObjective",
        )
        self.registry.register(
            backend="torch",
            ref="objective.mse",
            python="trainer.objectives:KeyedMSEObjective",
        )
        self.registry.register(
            backend="torch",
            ref="objective.classification_ce",
            python="trainer.objectives:KeyedCrossEntropyObjective",
        )

        # Metrics/evaluators
        self.registry.register(
            backend="torch",
            ref="metric.perplexity",
            python="benchmark.metrics:PerplexityMetric",
        )
        self.registry.register(
            backend="torch",
            ref="evaluator.latency",
            python="benchmark.evaluators:LatencyEvaluator",
        )
        self.registry.register(
            backend="torch",
            ref="evaluator.memory",
            python="benchmark.evaluators:MemoryEvaluator",
        )

    def run_experiment(
        self,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        *,
        dry_run: bool = False,
    ) -> Any:
        trainer = self.registry.build(target.trainer, backend=str(target.backend))
        if not hasattr(trainer, "run"):
            raise TypeError(
                f"Trainer component {target.trainer.ref!r} did not provide a run() method"
            )
        result = cast(Any, trainer).run(
            manifest=manifest,
            target=target,
            engine=self,
            dry_run=dry_run,
        )
        if dry_run:
            return result

        artifacts: dict[str, Path] = {}

        # Legacy benchmark suite (teacher/student).
        if target.benchmarks and isinstance(result, dict) and "teacher" in result and "student" in result:
            teacher = self._as_module(result["teacher"])
            student = self._as_module(result["student"])
            device = result.get("device", torch.device("cpu"))
            if not isinstance(device, torch.device):
                device = torch.device(str(device))
            suite = BenchmarkSuite(
                benchmarks=target.benchmarks,
                output_dir=str(
                    Path("artifacts")
                    / str(manifest.name or "experiment")
                    / str(target.name)
                    / datetime.now().strftime("%Y%m%d_%H%M%S")
                ),
                formats=["csv", "json", "png", "latex"],
            )
            metadata = ExperimentMetadata(
                name=str(target.name),
                timestamp=datetime.now().isoformat(),
                manifest_path=str(getattr(manifest, "name", "") or ""),
                teacher_checkpoint=str(
                    getattr(getattr(self._first_train(target), "teacher_ckpt", None), "__str__", lambda: "")()
                ),
                student_config=str(target.system.config.get("model", {}).get("type", "")),
                device=str(device),
                notes=str(getattr(manifest, "notes", "") or ""),
            )
            runner = BenchmarkRunner(suite, device, metadata)
            try:
                artifacts.update(runner.run(teacher, student))
            except Exception as e:
                logger.warning(f"Benchmarks failed: {e}")

        # Metrics/evaluators (best-effort).
        if target.metrics and isinstance(result, dict):
            models: dict[str, nn.Module] = {}
            if "teacher" in result and "student" in result:
                models["teacher"] = self._as_module(result["teacher"])
                models["student"] = self._as_module(result["student"])
            elif "system" in result:
                models["system"] = self._as_module(result["system"])

            device = result.get("device", torch.device("cpu"))
            if not isinstance(device, torch.device):
                device = torch.device(str(device))

            for spec in target.metrics:
                try:
                    metric = self.registry.build(spec, backend=str(target.backend))
                except Exception as e:
                    logger.warning(f"Failed to build metric {spec.ref}: {e}")
                    continue
                for name, model in models.items():
                    if hasattr(metric, "run"):
                        try:
                            m = cast(Any, metric).run(model=model, device=device, name=name)
                            logger.info(f"metric[{spec.ref}]({name})={m}")
                        except Exception as e:
                            logger.warning(f"Metric {spec.ref} failed for {name}: {e}")

        return artifacts

    @staticmethod
    def _as_module(obj: object) -> nn.Module:
        if isinstance(obj, nn.Module):
            return obj
        m = getattr(obj, "module", None)
        if isinstance(m, nn.Module):
            return m
        raise TypeError(f"Expected nn.Module-like object, got {type(obj).__name__}")

    @staticmethod
    def _first_train(target: ExperimentTargetConfig) -> object:
        for r in target.runs:
            if r.train is not None:
                return r.train
        return object()

