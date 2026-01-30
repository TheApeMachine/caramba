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
import json
import os
import subprocess
import sys


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
            ref="trainer.finetune_minimal",
            python="caramba.trainer.finetune_minimal:FinetuneMinimalTrainer",
        )
        self.registry.register(
            backend="torch",
            ref="trainer.finetune_unsloth",
            python="caramba.trainer.finetune_unsloth:FinetuneUnslothTrainer",
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
        self.registry.register(
            backend="torch",
            ref="trainer.ccl",
            python="caramba.trainer.ccl:CCLTrainer",
        )
        self.registry.register(
            backend="torch",
            ref="trainer.multi_checkpoint_compare",
            python="caramba.trainer.multi_checkpoint_compare:MultiCheckpointCompareTrainer",
        )
        self.registry.register(
            backend="torch",
            ref="trainer.surgery_compare",
            python="caramba.trainer.surgery_compare:SurgeryCompareTrainer",
        )
        self.registry.register(
            backend="torch",
            ref="trainer.surgery_export",
            python="caramba.trainer.surgery_export:SurgeryExportTrainer",
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
        self.registry.register(
            backend="torch",
            ref="dataset.hf_image_classification",
            python="caramba.data.hf_image_classification:HFImageClassificationDataset",
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
        self.registry.register(
            backend="torch",
            ref="system.ccl",
            python="caramba.ccl.system:CCLSystem",
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
        self.registry.register(
            backend="torch",
            ref="objective.none",
            python="caramba.trainer.objectives:ZeroObjective",
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
        # Result formats:
        # - Multi-checkpoint: {"models": dict[str, Module], "baseline_name": str, ...}
        # - Upcycle targets: {"teacher", "student"}
        # - Standard scratch: {"system"}; treat that as "student" for benchmarks.
        if target.benchmarks and isinstance(result, dict):
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
                student_config=str(target.system.config.get("model", {}).get("type", "") if hasattr(target, "system") and target.system else ""),
                device=str(device),
                notes=str(getattr(manifest, "notes", "") or ""),
            )

            # Multi-model isolation format: {"checkpoint_specs": [...], ...}
            if "checkpoint_specs" in result and isinstance(result["checkpoint_specs"], list):
                from benchmark.multi_model_runner import MultiModelBenchmarkRunner
                from carmath import weight_dtype

                specs = [s for s in result["checkpoint_specs"] if isinstance(s, dict) and "name" in s]
                specs_by_name = {str(s["name"]): s for s in specs}
                model_names = sorted(specs_by_name.keys())
                baseline_name = result.get("baseline_name")

                dtype_str = str(result.get("dtype", "auto")).lower().strip()
                strict = bool(result.get("strict", True))
                unsafe_pickle_load = bool(result.get("unsafe_pickle_load", False))
                # Default to strongest isolation when supported by the trainer.
                process_isolation = bool(result.get("process_isolation", True))

                if process_isolation:
                    # Strongest isolation: run the entire multi-model suite in a subprocess.
                    job = {
                        "output_dir": str(suite.output_dir),
                        "suite": suite.model_dump(),
                        "metadata": {
                            "name": metadata.name,
                            "timestamp": metadata.timestamp,
                            "manifest_path": metadata.manifest_path,
                            "teacher_checkpoint": metadata.teacher_checkpoint,
                            "student_config": metadata.student_config,
                            "device": metadata.device,
                            "notes": metadata.notes,
                        },
                        "checkpoint_specs": list(specs),
                        "baseline_name": baseline_name,
                        "device": str(device),
                        "dtype": dtype_str,
                        "strict": strict,
                        "unsafe_pickle_load": unsafe_pickle_load,
                    }

                    output_dir = Path(str(suite.output_dir))
                    output_dir.mkdir(parents=True, exist_ok=True)
                    job_path = output_dir / "_process_isolation_job.json"
                    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")

                    repo_root = Path(__file__).resolve().parents[2]
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(repo_root) + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
                    cmd = [sys.executable, "-m", "benchmark.isolated_multi_model_run", "--job", str(job_path)]
                    logger.info(f"process-isolation: spawning {' '.join(cmd)}")
                    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)

                    index_path = Path(str(suite.output_dir)) / "artifacts_index.json"
                    if index_path.exists():
                        try:
                            index = json.loads(index_path.read_text(encoding="utf-8"))
                            if isinstance(index, dict):
                                for k, v in index.items():
                                    try:
                                        artifacts[str(k)] = Path(str(v))
                                    except Exception:
                                        pass
                        except Exception as e:
                            logger.warning(f"Failed to read artifacts_index.json: {e!r}")
                else:
                    # In-process load/unload isolation.
                    from model import Model
                    from trainer.checkpoint_compare import (
                        _lower_and_validate_model_config,
                        _safe_load_checkpoint,
                    )
                    import gc

                    dt = weight_dtype(device, dtype_str if dtype_str != "auto" else "auto")

                    def _gc_and_empty_cache() -> None:
                        try:
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

                    def load_model(model_name: str) -> nn.Module:
                        spec = specs_by_name.get(str(model_name))
                        if spec is None:
                            raise KeyError(f"Unknown model_name={model_name!r}")
                        ckpt_path = Path(str(spec["checkpoint"]))
                        if not ckpt_path.exists():
                            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
                        cfg = _lower_and_validate_model_config(spec["model_config"])
                        m = Model(cfg).to(device=device, dtype=dt)
                        sd = _safe_load_checkpoint(ckpt_path, unsafe_pickle_load=unsafe_pickle_load)
                        m.load_state_dict(sd, strict=strict)
                        m.eval()
                        return self._as_module(m)

                    def unload_model(m: nn.Module) -> None:
                        try:
                            m.to("cpu")
                        except Exception:
                            pass
                        try:
                            del m
                        except Exception:
                            pass
                        _gc_and_empty_cache()

                    runner = MultiModelBenchmarkRunner(suite, device, metadata, baseline_name)
                    try:
                        artifacts.update(runner.run_isolated(model_names, load_model, unload_model))
                    except Exception as e:
                        logger.warning(f"Multi-model isolated benchmarks failed: {e}")
                        import traceback
                        traceback.print_exc()

            # Multi-model eager format: {"models": dict[str, Module], "baseline_name": str}
            elif "models" in result and isinstance(result["models"], dict):
                from benchmark.multi_model_runner import MultiModelBenchmarkRunner

                models_dict = result["models"]
                # Convert to nn.Module dict
                models: dict[str, nn.Module] = {}
                for name, m in models_dict.items():
                    if m is not None:  # Skip None (dry_run placeholders)
                        models[name] = self._as_module(m)

                baseline_name = result.get("baseline_name")
                runner = MultiModelBenchmarkRunner(suite, device, metadata, baseline_name)
                try:
                    artifacts.update(runner.run(models))
                except Exception as e:
                    logger.warning(f"Multi-model benchmarks failed: {e}")
                    import traceback
                    traceback.print_exc()

            # Legacy 2-model format: {"teacher", "student"} or {"system"}
            elif ("teacher" in result and "student" in result) or ("system" in result):
                teacher = self._as_module(result["teacher"]) if "teacher" in result else None
                student = (
                    self._as_module(result["student"])
                    if "student" in result
                    else self._as_module(result["system"])
                )
                runner = BenchmarkRunner(suite, device, metadata)
                try:
                    artifacts.update(runner.run(teacher, student))
                except Exception as e:
                    logger.warning(f"Benchmarks failed: {e}")

        # Metrics/evaluators.
        if target.metrics and isinstance(result, dict):
            metrics_models: dict[str, nn.Module] = {}
            if "teacher" in result and "student" in result:
                metrics_models["teacher"] = self._as_module(result["teacher"])
                metrics_models["student"] = self._as_module(result["student"])
            elif "system" in result:
                metrics_models["system"] = self._as_module(result["system"])

            device = result.get("device", torch.device("cpu"))
            if not isinstance(device, torch.device):
                device = torch.device(str(device))

            for spec in target.metrics:
                try:
                    metric = self.registry.build(spec, backend=str(target.backend))
                except Exception as e:
                    logger.warning(f"Failed to build metric {spec.ref}: {e}")
                    continue
                for name, model in metrics_models.items():
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

