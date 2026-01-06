"""Checkpoint-compare trainer (evaluation-only).

Loads two checkpoints (teacher vs student) with potentially different model
configs, then returns them to the engine so the standard benchmark suite can
run and write artifacts.

This is meant for paper workflows where you already have paired checkpoints
(e.g., baseline vs DBA) and want a manifest-driven, reproducible compare run.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, cast

import torch

from caramba.carmath import weight_dtype
from caramba.compiler.lower import Lowerer
from caramba.compiler.validate import Validator
from caramba.config.model import ModelConfig
from caramba.console import logger
from caramba.model import Model


def _lower_and_validate_model_config(payload: dict[str, Any]) -> ModelConfig:
    cfg = ModelConfig.model_validate(payload)
    cfg = Lowerer().lower_model(cfg)
    Validator().validate_model_config(cfg)
    return cfg


def _extract_state_dict(obj: object) -> dict[str, torch.Tensor]:
    """Best-effort checkpoint container parsing.

    Supports:
    - raw state_dict: {param_name: Tensor, ...}
    - {"system_state_dict": ...}
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    - {"student_state_dict": ...} (legacy upcycle-eval shape)
    """
    if isinstance(obj, dict):
        # Raw state dict (most common for safetensors / some .pt dumps).
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return cast(dict[str, torch.Tensor], obj)

        for k in ("system_state_dict", "model_state_dict", "state_dict", "student_state_dict"):
            if k in obj:
                sd = obj.get(k)
                if isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values()):
                    return cast(dict[str, torch.Tensor], sd)
                raise TypeError(f"Checkpoint key {k!r} exists but is not a tensor state_dict")

    raise TypeError(
        "Unsupported checkpoint format. Expected a state_dict-like dict or a dict containing "
        "one of: system_state_dict, model_state_dict, state_dict, student_state_dict."
    )


def _safe_load_checkpoint(
    path: Path, *, unsafe_pickle_load: bool
) -> dict[str, torch.Tensor]:
    """Load a checkpoint with a safe default, optional unsafe fallback."""
    # SECURITY: prefer weights_only=True to avoid arbitrary code execution.
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, pickle.UnpicklingError, ValueError, EOFError) as e:
        logger.warning(
            "Secure checkpoint load failed with weights_only=True. "
            "This checkpoint likely contains pickled objects and requires an unsafe load.\n"
            "Only proceed if you FULLY trust this checkpoint file.\n"
            f"Error: {e}\n"
            "To opt in, set trainer config unsafe_pickle_load=true in the manifest and retry."
        )
        if not unsafe_pickle_load:
            raise
        logger.warning("unsafe_pickle_load=true is set; reloading with weights_only=False (UNSAFE).")
        obj = torch.load(path, map_location="cpu", weights_only=False)

    return _extract_state_dict(obj)


class CheckpointCompareTrainer:
    """Load two checkpoints (teacher/student) and return modules for benchmarks."""

    def __init__(
        self,
        *,
        teacher_ckpt: str,
        student_ckpt: str,
        # Teacher model config payload (dict). Student model config can be provided
        # here or taken from target.system.config.model.
        teacher_model: dict[str, Any],
        student_model: dict[str, Any] | None = None,
        device: str = "cpu",
        dtype: str = "auto",
        strict: bool = True,
        unsafe_pickle_load: bool = False,
    ) -> None:
        self.teacher_ckpt = str(teacher_ckpt)
        self.student_ckpt = str(student_ckpt)
        self.teacher_model = dict(teacher_model)
        self.student_model = dict(student_model) if student_model is not None else None
        self.device = torch.device(str(device))
        self.dtype = str(dtype).lower().strip()
        self.strict = bool(strict)
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

        # Student config fallback: use target.system.config.model (keeps manifests tidy).
        student_payload: dict[str, Any] | None = None
        if self.student_model is not None:
            student_payload = dict(self.student_model)
        else:
            if getattr(target, "system", None) is None or target.system.ref != "system.language_model":
                raise ValueError(
                    "CheckpointCompareTrainer requires either student_model=... in trainer config "
                    "or target.system.ref=system.language_model with system.config.model"
                )
            mp = target.system.config.get("model", None)
            if not isinstance(mp, dict):
                raise ValueError("system.language_model requires system.config.model (dict)")
            student_payload = dict(mp)

        teacher_cfg = _lower_and_validate_model_config(self.teacher_model)
        student_cfg = _lower_and_validate_model_config(cast(dict[str, Any], student_payload))

        dt = weight_dtype(self.device, self.dtype if self.dtype != "auto" else "auto")
        logger.header("Checkpoint Compare", f"device={self.device.type} dtype={dt}")

        teacher = Model(teacher_cfg).to(device=self.device, dtype=dt)
        student = Model(student_cfg).to(device=self.device, dtype=dt)

        # ---- load weights ----
        t_path = Path(self.teacher_ckpt)
        s_path = Path(self.student_ckpt)
        if not t_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {t_path}")
        if not s_path.exists():
            raise FileNotFoundError(f"Student checkpoint not found: {s_path}")

        logger.info(f"Loading teacher checkpoint: {t_path}")
        t_sd = _safe_load_checkpoint(t_path, unsafe_pickle_load=self.unsafe_pickle_load)
        t_res = teacher.load_state_dict(t_sd, strict=bool(self.strict))
        missing, unexpected = t_res
        if missing or unexpected:
            logger.warning(
                f"Teacher load_state_dict: missing={len(missing)} unexpected={len(unexpected)}"
            )

        logger.info(f"Loading student checkpoint: {s_path}")
        s_sd = _safe_load_checkpoint(s_path, unsafe_pickle_load=self.unsafe_pickle_load)
        s_res = student.load_state_dict(s_sd, strict=bool(self.strict))
        missing, unexpected = s_res
        if missing or unexpected:
            logger.warning(
                f"Student load_state_dict: missing={len(missing)} unexpected={len(unexpected)}"
            )

        teacher.eval()
        student.eval()

        return {
            "teacher": teacher,
            "student": student,
            "device": self.device,
            "checkpoint_dir": str(Path(getattr(manifest, "artifacts_dir", "artifacts")) / "benchmarks"),
        }

