"""Default initializer implementation for Upcycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from config.initializer import DefaultInitializerConfig
from config.model import ModelConfig
from config.train import TrainConfig
from console import logger
from loader.checkpoint import CheckpointLoader
from loader.hf import HFLoader
from loader.llama_upcycle import LlamaUpcycle
from model import Model
from trainer.upcycle_init_context import UpcycleInitContext


def _make_teacher_model_config(model_config: ModelConfig) -> ModelConfig:
    """Teacher must use standard attention so pretrained weights load correctly."""

    def rewrite_node_dict(node_dict: dict[str, Any]) -> dict[str, Any]:
        if node_dict.get("type") == "AttentionLayer":
            node_dict = node_dict.copy()
            node_dict["mode"] = "standard"
            # Student-only bottleneck knobs must not affect teacher shapes.
            node_dict.pop("attn_dim", None)
            node_dict.pop("sem_dim", None)
            node_dict.pop("geo_dim", None)
            node_dict.pop("decoupled_gate", None)
            node_dict.pop("decoupled_gate_dynamic", None)
            # Teacher should prefer the largest fused kernels available; training-time
            # chunk/window settings are primarily for student memory control.
            node_dict.pop("q_chunk", None)
            node_dict.pop("local_window", None)
            return node_dict

        if "layers" in node_dict and isinstance(node_dict["layers"], list):
            node_dict = node_dict.copy()
            node_dict["layers"] = [
                rewrite_node_dict(layer) if isinstance(layer, dict) else layer
                for layer in node_dict["layers"]
            ]
        return node_dict

    teacher_data = model_config.model_dump()
    teacher_data["topology"] = rewrite_node_dict(teacher_data["topology"])
    return ModelConfig.model_validate(teacher_data)


class DefaultInitializer:
    def __init__(self, config: DefaultInitializerConfig) -> None:
        self.config = config

    def init_models(self, train: TrainConfig, ctx: UpcycleInitContext) -> tuple[nn.Module, nn.Module]:
        if train.teacher_ckpt is None:
            raise ValueError("train.teacher_ckpt is required for upcycle.")
        raw_model = getattr(ctx.manifest, "model", None)
        if raw_model is None:
            raise ValueError(
                "Upcycle requires ctx.manifest.model (a ModelConfig or dict payload), but it was missing."
            )
        model_cfg = (
            raw_model
            if isinstance(raw_model, ModelConfig)
            else ModelConfig.model_validate(raw_model)
        )

        logger.header("Model Initialization")
        logger.info(f"Loading teacher checkpoint: {train.teacher_ckpt}")
        ckpt_path = self._resolve_teacher_ckpt(str(train.teacher_ckpt))
        state_dict = CheckpointLoader().load(ckpt_path)
        logger.success(f"Loaded checkpoint with {len(state_dict)} keys")

        logger.info("Building teacher model (standard attention)...")
        teacher_cfg = _make_teacher_model_config(model_cfg)
        teacher = Model(teacher_cfg).to(device=ctx.device, dtype=ctx.dtype)
        logger.success("Teacher model ready")

        logger.info("Building student model (manifest attention)...")
        student = Model(model_cfg).to(device=ctx.device, dtype=ctx.dtype)
        logger.success("Student model ready")

        logger.info("Applying weights to teacher...")
        LlamaUpcycle(teacher, state_dict).apply()

        logger.info("Applying upcycle surgery to student...")
        dba_init = str(getattr(train, "dba_init", "svd"))
        LlamaUpcycle(student, state_dict, dba_init=dba_init).apply()
        logger.success("Weight transfer complete")

        if ctx.dist_ctx is not None:
            logger.info("Wrapping student for distributed training...")
            student = ctx.dist_ctx.wrap_model(student)

        if bool(ctx.runtime_plan.compile) and hasattr(torch, "compile"):
            if ctx.device.type in ("cuda", "mps"):
                logger.info(
                    f"Compiling student with torch.compile (mode={ctx.runtime_plan.compile_mode})..."
                )
                try:
                    compiled = torch.compile(student, mode=ctx.runtime_plan.compile_mode)  # type: ignore[call-arg]
                    student = cast(nn.Module, compiled)
                    logger.success("torch.compile applied")
                except Exception as e:
                    logger.warning(f"torch.compile failed, continuing without: {e}")
            else:
                logger.warning(f"torch.compile not supported on {ctx.device.type}, skipping")

        teacher.eval()
        logger.success("Initialization complete")

        if bool(getattr(train, "activation_checkpointing", False)):
            threshold = float(getattr(train, "activation_checkpoint_threshold_mb", 0.0))
            for m in student.modules():
                if hasattr(m, "activation_checkpointing"):
                    try:
                        setattr(m, "activation_checkpointing", True)
                        setattr(m, "activation_checkpoint_threshold_mb", threshold)
                    except Exception:
                        pass

        return teacher, student

    @staticmethod
    def _resolve_teacher_ckpt(ckpt: str) -> Path:
        if ckpt.startswith("hf://"):
            return HFLoader(repo_id=ckpt[5:]).load()
        return Path(ckpt)

