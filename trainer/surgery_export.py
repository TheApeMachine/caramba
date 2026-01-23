"""HF teacher → DBA surgery export trainer (evaluation-only).

Purpose
-------
Create a *checkpoint pair* suitable for the existing paper-style benchmark flow:

- baseline: standard attention model loaded from HF teacher weights
- surgery:  DBA (decoupled) attention model, initialized via a simple DBA init policy

Both checkpoints are written as raw PyTorch state_dicts so they can be consumed by:
  - trainer.checkpoint_compare
  - trainer.multi_checkpoint_compare

This trainer does *no training*.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class SurgeryExportPaths:
    baseline: Path
    surgery: Path


class SurgeryExportTrainer:
    """Export baseline + surgically initialized DBA checkpoint pair."""

    def __init__(
        self,
        *,
        teacher_ckpt: str,
        # Model config payloads (dict). If teacher_model is omitted, we derive it
        # from student_model by forcing standard attention.
        student_model: dict[str, Any],
        teacher_model: dict[str, Any] | None = None,
        # DBA initialization policy (AdapterStateDictTransformer): svd|random|fresh
        dba_init: str = "fresh",
        # Output directory (created if missing)
        output_dir: str = "artifacts/surgery/llama32_1b",
        # Runtime
        device: str = "cpu",
        dtype: str = "auto",
        gate_init_bias: float | None = None,
        out_proj_init_std: float | None = None,
    ) -> None:
        self.teacher_ckpt = str(teacher_ckpt)
        self.student_model = dict(student_model)
        self.teacher_model = dict(teacher_model) if teacher_model is not None else None
        self.dba_init = str(dba_init).lower().strip()
        self.output_dir = str(output_dir)
        self.device = torch.device(str(device))
        self.dtype = str(dtype).lower().strip()
        self.gate_init_bias = gate_init_bias
        self.out_proj_init_std = out_proj_init_std

    def run(
        self,
        *,
        manifest: Any,
        target: Any,
        engine: Any,  # unused; parity with other trainers
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        if dry_run:
            return {
                "baseline_ckpt": str(Path(self.output_dir) / "baseline.pt"),
                "surgery_ckpt": str(Path(self.output_dir) / "surgery.pt"),
            }

        dt = weight_dtype(self.device, self.dtype if self.dtype != "auto" else "auto")
        logger.header(
            "Surgery Export",
            f"device={self.device.type} dtype={dt} dba_init={self.dba_init}",
        )

        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = SurgeryExportPaths(
            baseline=out_dir / "baseline.pt",
            surgery=out_dir / "surgery.pt",
        )

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

        # Safety: ensure teacher uses standard attention even if teacher_model was provided.
        teacher_cfg = _make_teacher_model_config(teacher_cfg)

        # ---- Build models ----
        baseline = Model(teacher_cfg).to(device=self.device, dtype=dt)
        surgery = Model(student_cfg).to(device=self.device, dtype=dt)

        # ---- Apply weights ----
        logger.info("Applying weights to baseline (standard attention)...")
        AdapterStateDictTransformer.llama(dba_init="svd").apply(model=baseline, state_dict=teacher_state)
        baseline.eval()

        logger.info("Applying weights to surgery model (manifest attention)...")
        AdapterStateDictTransformer.llama(
            dba_init=self.dba_init,
            gate_init_bias=self.gate_init_bias,
            out_proj_init_std=self.out_proj_init_std,
        ).apply(model=surgery, state_dict=teacher_state)
        surgery.eval()

        # ---- Save checkpoints (raw state_dict on CPU for portability) ----
        logger.info(f"Saving baseline state_dict: {paths.baseline}")
        self._save_state_dict(paths.baseline, baseline)
        logger.info(f"Saving surgery state_dict: {paths.surgery}")
        self._save_state_dict(paths.surgery, surgery)
        logger.success("Export complete")

        return {
            "baseline_ckpt": str(paths.baseline),
            "surgery_ckpt": str(paths.surgery),
            "device": self.device,
        }

    @staticmethod
    def _load_teacher_state(ckpt: str) -> dict[str, torch.Tensor]:
        p = Path(ckpt)
        if ckpt.startswith("hf://"):
            p = HFLoader(repo_id=ckpt[5:]).load()
        state = CheckpointBuilder().load(p)
        return cast(dict[str, torch.Tensor], state)

    @staticmethod
    def _save_state_dict(path: Path, model: nn.Module) -> None:
        sd = model.state_dict()
        sd_cpu = {k: v.detach().to(device="cpu") for k, v in sd.items()}
        # SECURITY: saving tensors only; safe to load with weights_only=True.
        torch.save(sd_cpu, path)

