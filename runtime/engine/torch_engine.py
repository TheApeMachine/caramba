"""PyTorch execution engine.

This is the first concrete engine. Other engines (JAX/sklearn/etc.) can be
added later without changing the manifest schema: only registry mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from caramba.config.manifest import Manifest
from caramba.config.target import ExperimentTargetConfig
from caramba.runtime.registry import ComponentRegistry
from caramba.benchmark.artifacts import ExperimentMetadata
from caramba.benchmark.runner import BenchmarkRunner
from caramba.config.benchmark import BenchmarkSuite
from caramba.console import logger
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
            python="caramba.trainer.upcycle:UpcycleTrainer",
        )
        self.registry.register(
            backend="torch",
            ref="trainer.upcycle_eval",
            python="caramba.trainer.upcycle_eval:UpcycleEvalTrainer",
        )
        self.registry.register(
            backend="torch",
            ref="trainer.standard",
            python="caramba.trainer.standard:StandardTrainer",
        )
        self.registry.register(
            backend="torch",
            ref="trainer.checkpoint_compare",
            python="caramba.trainer.checkpoint_compare:CheckpointCompareTrainer",
        )
        self.registry.register(
            backend="torch",
            ref="trainer.gradient_isolation",
            python="caramba.trainer.gradient_isolation:GradientIsolationTrainer",
        )
        self.registry.register(
            backend="torch",
            ref="trainer.diffusion_codegen",
            python="caramba.trainer.diffusion_codegen.trainer:DiffusionCodegenTrainer",
        )

        # Datasets
        self.registry.register(
            backend="torch",
            ref="dataset.tokens",
            python="caramba.data.tokens:TokenDataset",
        )
        self.registry.register(
            backend="torch",
            ref="dataset.mosaic_memory_curriculum",
            python="caramba.data.mosaic_synth:MosaicMemoryCurriculumDataset",
        )
        self.registry.register(
            backend="torch",
            ref="dataset.mosaic_event_traces",
            python="caramba.data.event_trace:MosaicEventTraceDataset",
        )
        self.registry.register(
            backend="torch",
            ref="dataset.icl_rule_induction",
            python="caramba.data.icl_rule:IclRuleInductionDataset",
        )
        self.registry.register(
            backend="torch",
            ref="dataset.random_tokens",
            python="data.random_tokens:RandomTokenDataset",
        )
        self.registry.register(
            backend="torch",
            ref="dataset.npy_supervised",
            python="caramba.data.npy_supervised:NpySupervisedDataset",
        )
        self.registry.register(
            backend="torch",
            ref="dataset.graph_npy",
            python="caramba.data.graph_npy:GraphNpyDataset",
        )
        self.registry.register(
            backend="torch",
            ref="dataset.diffusion_vector",
            python="caramba.data.diffusion_vector:DiffusionVectorDataset",
        )
        self.registry.register(
            backend="torch",
            ref="dataset.codegen_chunks",
            python="caramba.data.code_chunks:CodeChunksDataset",
        )
        self.registry.register(
            backend="torch",
            ref="dataset.tensors",
            python="caramba.data.tensors:TensorFilesDataset",
        )

        # Systems
        self.registry.register(
            backend="torch",
            ref="system.language_model",
            python="caramba.model.language_model_system:LanguageModelSystem",
        )
        self.registry.register(
            backend="torch",
            ref="system.generic",
            python="caramba.model.generic_system:GenericSystem",
        )
        self.registry.register(
            backend="torch",
            ref="system.mlp_classifier",
            python="caramba.model.mlp_classifier_system:MLPClassifierSystem",
        )
        self.registry.register(
            backend="torch",
            ref="system.gcn",
            python="caramba.model.gcn_system:GCNSystem",
        )
        self.registry.register(
            backend="torch",
            ref="system.diffusion_denoiser",
            python="caramba.model.diffusion_denoiser_system:DiffusionDenoiserSystem",
        )
        self.registry.register(
            backend="torch",
            ref="system.diffusion_codegen",
            python="caramba.model.diffusion_codegen_system:DiffusionCodegenSystem",
        )

        # Objectives
        self.registry.register(
            backend="torch",
            ref="objective.next_token_ce",
            python="caramba.trainer.objectives:NextTokenCrossEntropyObjective",
        )
        self.registry.register(
            backend="torch",
            ref="objective.next_token_ce_chunked",
            python="caramba.trainer.objectives:NextTokenCrossEntropyChunkedObjective",
        )
        self.registry.register(
            backend="torch",
            ref="objective.mosaic_next_token_aux",
            python="caramba.trainer.objectives:MosaicNextTokenWithAuxObjective",
        )
        self.registry.register(
            backend="torch",
            ref="objective.mosaic_event_prediction",
            python="caramba.trainer.objectives:MosaicEventPrediction",
        )
        self.registry.register(
            backend="torch",
            ref="objective.mse",
            python="caramba.trainer.objectives:KeyedMSEObjective",
        )
        self.registry.register(
            backend="torch",
            ref="objective.classification_ce",
            python="caramba.trainer.objectives:KeyedCrossEntropyObjective",
        )

        # Metrics/evaluators
        self.registry.register(
            backend="torch",
            ref="metric.perplexity",
            python="caramba.benchmark.metrics:PerplexityMetric",
        )
        self.registry.register(
            backend="torch",
            ref="evaluator.latency",
            python="caramba.benchmark.evaluators:LatencyEvaluator",
        )
        self.registry.register(
            backend="torch",
            ref="evaluator.memory",
            python="caramba.benchmark.evaluators:MemoryEvaluator",
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

        # Optional benchmark suite.
        #
        # - Upcycle targets return {"teacher", "student"}.
        # - Standard scratch targets return {"system"}; treat that as "student" for benchmarks.
        if target.benchmarks and isinstance(result, dict) and (
            ("teacher" in result and "student" in result) or ("system" in result)
        ):
            teacher = self._as_module(result["teacher"]) if "teacher" in result else None
            student = (
                self._as_module(result["student"])
                if "student" in result
                else self._as_module(result["system"])
            )
            device = result.get("device", torch.device("cpu"))
            if not isinstance(device, torch.device):
                device = torch.device(str(device))
            suite = BenchmarkSuite(
                benchmarks=target.benchmarks,
                output_dir=str(
                    Path(str(getattr(manifest, "artifacts_dir", "artifacts") or "artifacts"))
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

