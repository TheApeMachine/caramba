"""HF teacher → DBA surgery compare trainer (evaluation-only).

This trainer mirrors the *benchmark harness* used by the DBA paper manifests
(perplexity/latency/memory/behavior/etc.), but instead of loading two (or N)
pretrained checkpoints from disk, it:

- loads a HuggingFace teacher checkpoint (e.g. Llama 3.2 1B)
- builds a baseline (standard attention) model and a DBA (decoupled) model
- applies the teacher weights to both via AdapterStateDictTransformer
  - baseline: standard attention load
  - DBA: decoupled attention load with configurable DBA init policy

It then returns {"models": {...}, "baseline_name": "..."} so the Torch engine
uses MultiModelBenchmarkRunner, matching the baseline/DBA comparison flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from adapter.state_dict import AdapterStateDictTransformer
from carmath import weight_dtype
from config.model import ModelConfig
from console import logger
from loader.checkpoint import CheckpointBuilder
from loader.hf import HFLoader
from model import Model
from trainer.initializers.default import _make_teacher_model_config


class SurgeryCompareTrainer:
    """Build teacher + surgically-initialized DBA student; benchmark both."""

    def __init__(
        self,
        *,
        teacher_ckpt: str,
        # Model config payloads (dict). If teacher_model is omitted, we derive it
        # from student_model by forcing standard attention (Llama parity).
        student_model: dict[str, Any],
        teacher_model: dict[str, Any] | None = None,
        # Output naming (for MultiModelBenchmarkRunner). Defaults preserve older manifests.
        teacher_name: str = "baseline",
        student_name: str = "student",
        baseline_name: str | None = None,
        # DBA initialization policy (AdapterStateDictTransformer): svd|random|fresh
        dba_init: str = "fresh",
        # Runtime
        device: str = "cpu",
        dtype: str = "auto",
    ) -> None:
        self.teacher_ckpt = str(teacher_ckpt)
        self.student_model = dict(student_model)
        self.teacher_model = dict(teacher_model) if teacher_model is not None else None
        self.teacher_name = str(teacher_name)
        self.student_name = str(student_name)
        self.baseline_name = str(baseline_name) if baseline_name is not None else None
        self.dba_init = str(dba_init).lower().strip()
        self.device = torch.device(str(device))
        self.dtype = str(dtype).lower().strip()

    def run(
        self,
        *,
        manifest: Any,
        target: Any,
        engine: Any,  # unused; parity with other trainers
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        if dry_run:
            base = self.baseline_name or self.teacher_name
            return {"models": {self.teacher_name: None, self.student_name: None}, "baseline_name": base}

        dt = weight_dtype(self.device, self.dtype if self.dtype != "auto" else "auto")
        logger.header("Surgery Compare", f"device={self.device.type} dtype={dt} dba_init={self.dba_init}")

        # ---- Load teacher weights (HF or local path) ----
        logger.info(f"Loading teacher checkpoint: {self.teacher_ckpt}")
        teacher_state = self._load_teacher_state(self.teacher_ckpt)
        logger.success(f"Loaded teacher state with {len(teacher_state)} keys")

        # ---- Build model configs ----
        student_cfg = ModelConfig.model_validate(self.student_model)
        if self.teacher_model is not None:
            teacher_cfg = ModelConfig.model_validate(self.teacher_model)
        else:
            teacher_cfg = _make_teacher_model_config(student_cfg)

        # Safety: ensure teacher uses standard attention even if a teacher_model was provided.
        # This avoids subtle shape mismatches when loading HF Llama weights.
        teacher_cfg = _make_teacher_model_config(teacher_cfg)

        # ---- Build models ----
        baseline = Model(teacher_cfg).to(device=self.device, dtype=dt)
        student = Model(student_cfg).to(device=self.device, dtype=dt)

        # ---- Apply weights ----
        # Baseline teacher: standard attention load using the llama schema.
        logger.info("Applying weights to baseline (standard attention)...")
        AdapterStateDictTransformer.llama(dba_init="svd").apply(model=baseline, state_dict=teacher_state)
        baseline.eval()

        # Student: decoupled attention (if configured) and DBA init policy.
        logger.info("Applying weights to student (DBA/manifest attention)...")
        AdapterStateDictTransformer.llama(dba_init=self.dba_init).apply(model=student, state_dict=teacher_state)
        student.eval()

        if not self.teacher_name or not self.student_name:
            raise ValueError("SurgeryCompareTrainer requires non-empty teacher_name and student_name.")
        if self.teacher_name == self.student_name:
            raise ValueError("SurgeryCompareTrainer requires teacher_name != student_name.")
        base_name = self.baseline_name or self.teacher_name

        return {
            "models": {
                self.teacher_name: cast(nn.Module, baseline),
                self.student_name: cast(nn.Module, student),
            },
            "baseline_name": base_name,
            "device": self.device,
            # Torch engine will write artifacts under manifest.name/target.name/timestamp.
            "checkpoint_dir": str(Path(getattr(manifest, "artifacts_dir", "artifacts")) / "benchmarks"),
        }

    @staticmethod
    def _load_teacher_state(ckpt: str) -> dict[str, torch.Tensor]:
        p = Path(ckpt)
        if ckpt.startswith("hf://"):
            p = HFLoader(repo_id=ckpt[5:]).load()
        state = CheckpointBuilder().load(p)
        return cast(dict[str, torch.Tensor], state)

