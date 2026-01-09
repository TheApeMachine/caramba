"""Evaluation-only trainer for upcycled models (checkpoint reuse).

This trainer exists to support serious-paper workflows where we want to:
  - train a student once (exp_svd)
  - reuse the resulting checkpoint for multiple inference-only sweeps (exp_quant)

Unlike UpcycleTrainer, this class performs *no training*. It builds the teacher
and student, loads weights, and returns them so the engine can run benchmarks.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from caramba.adapter.state_dict import AdapterStateDictTransformer
from caramba.carmath import weight_dtype
from caramba.config.model import ModelConfig
from caramba.console import logger
from caramba.loader.checkpoint import CheckpointBuilder
from caramba.loader.hf import HFLoader
from caramba.model import Model
from caramba.trainer.initializers.default import _make_teacher_model_config


class UpcycleEvalTrainer:
    """Load teacher + student from checkpoints for benchmark-only runs."""

    def __init__(
        self,
        *,
        # Teacher (HF checkpoint or local path, same format as train.teacher_ckpt)
        teacher_ckpt: str,
        # Student checkpoint saved by DefaultCheckPointer (contains student_state_dict)
        student_ckpt: str,
        # Device/dtype for evaluation (e.g. "mps", "cuda", "cpu"; dtype "auto|float16|float32")
        device: str = "cpu",
        dtype: str = "auto",
        # If True, allow unsafe pickle loading of checkpoints when weights_only fails.
        # This MUST be set via manifest config; there are no environment-variable overrides.
        unsafe_pickle_load: bool = False,
    ) -> None:
        self.teacher_ckpt = str(teacher_ckpt)
        self.student_ckpt = str(student_ckpt)
        self.device = torch.device(str(device))
        self.dtype = str(dtype).lower().strip()
        self.unsafe_pickle_load = bool(unsafe_pickle_load)

    def run(
        self,
        *,
        manifest: Any,
        target: Any,
        engine: Any,  # unused; parity with other trainers
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        if dry_run:
            return None

        if getattr(target, "system", None) is None or target.system.ref != "system.language_model":
            raise ValueError("UpcycleEvalTrainer requires system.ref=system.language_model")

        model_payload = target.system.config.get("model", None)
        if not isinstance(model_payload, dict):
            raise ValueError("system.language_model requires system.config.model (dict)")
        model_cfg = ModelConfig.model_validate(model_payload)

        dt = weight_dtype(self.device, self.dtype if self.dtype != "auto" else "auto")
        logger.header("Upcycle Eval", f"device={self.device.type} dtype={dt}")

        # ---- teacher ----
        logger.info(f"Loading teacher checkpoint: {self.teacher_ckpt}")
        teacher_state = self._load_teacher_state(self.teacher_ckpt)
        teacher_cfg = _make_teacher_model_config(model_cfg)
        teacher = Model(teacher_cfg).to(device=self.device, dtype=dt)
        AdapterStateDictTransformer.llama(dba_init="svd").apply(model=teacher, state_dict=teacher_state)
        teacher.eval()

        # ---- student ----
        student = Model(model_cfg).to(device=self.device, dtype=dt)
        self._load_student_checkpoint(student, self.student_ckpt)
        student.eval()

        return {
            "teacher": teacher,
            "student": student,
            "device": self.device,
            "checkpoint_dir": str(Path("runs") / str(getattr(target, "name", "target"))),
        }

    @staticmethod
    def _load_teacher_state(ckpt: str) -> dict[str, torch.Tensor]:
        p = Path(ckpt)
        if ckpt.startswith("hf://"):
            p = HFLoader(repo_id=ckpt[5:]).load()
        state = CheckpointBuilder().load(p)
        return cast(dict[str, torch.Tensor], state)

    def _load_student_checkpoint(self, student: nn.Module, ckpt_path: str) -> None:
        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"Student checkpoint not found: {p}")
        logger.info(f"Loading student checkpoint: {p}")
        # SECURITY: prefer weights_only=True to avoid executing arbitrary code via pickle.
        # Only load pickle-based checkpoints (weights_only=False) if you FULLY trust the source.
        try:
            obj = torch.load(p, map_location="cpu", weights_only=True)
        except (RuntimeError, pickle.UnpicklingError, ValueError, EOFError) as e:
            logger.warning(
                "Secure checkpoint load failed with weights_only=True. "
                "This checkpoint likely contains pickled objects and requires an unsafe load.\n"
                "Only proceed if you FULLY trust this checkpoint file.\n"
                f"Error: {e}\n"
                "To opt in, set trainer config unsafe_pickle_load=true in the manifest and retry."
            )
            if self.unsafe_pickle_load:
                logger.warning(
                    "unsafe_pickle_load=true is set; reloading with weights_only=False (UNSAFE)."
                )
                obj = torch.load(p, map_location="cpu", weights_only=False)
            else:
                raise
        if not isinstance(obj, dict) or "student_state_dict" not in obj:
            raise TypeError("Student checkpoint must be a dict with key 'student_state_dict'")
        sd = obj["student_state_dict"]
        if not isinstance(sd, dict):
            raise TypeError("student_state_dict must be a dict")
        student.load_state_dict(sd)  # type: ignore[arg-type]
        logger.success("Student checkpoint loaded")

