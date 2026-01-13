"""Optimizer builder for standard training loops."""

from __future__ import annotations

import torch

from caramba.config.train import TrainConfig
from caramba.console import logger
from caramba.kernel.adamw_master import AdamWMaster
from caramba.kernel.lion import Lion


def build_optimizer(
    *,
    system: object,
    train: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.optim.Optimizer:
    if not hasattr(system, "parameters"):
        raise TypeError("System component does not expose parameters()")

    params = system.parameters()  # type: ignore[attr-defined]
    opt_name = str(getattr(train, "optimizer", "adamw")).lower()
    weight_decay = float(getattr(train, "weight_decay", 0.0))
    fused_opt = bool(getattr(train, "fused_optimizer", True))
    lr = float(train.lr)

    if opt_name in ("adamw", "adam"):
        if opt_name == "adamw":
            if fused_opt:
                # CPU has no fused optimizer backend; make this explicit.
                if device.type == "cpu":
                    logger.warning(
                        "AdamW fused optimizer requested on CPU; falling back to torch.optim.AdamW. "
                        "Set train.fused_optimizer=false to silence this warning."
                    )
                    fused_opt = False

                use_master = (device.type == "mps" and dtype in (torch.float16, torch.float32)) or (
                    device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
                )
                if use_master:
                    logger.info(
                        f"optimizer=adamw fused=true backend=adamw_master device={device.type} dtype={dtype}"
                    )
                    return AdamWMaster(
                        params,  # type: ignore[arg-type]
                        lr=lr,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=float(weight_decay),
                        fused=True,
                    )

                # On accelerators, this is a performance-critical request; fail loud.
                if device.type in ("cuda", "mps"):
                    raise RuntimeError(
                        "AdamW fused optimizer requested but unsupported for this device/dtype.\n"
                        f"device={device.type} dtype={dtype}\n"
                        "Supported: MPS fp16/fp32, CUDA fp16/bf16."
                    )

            logger.info(f"optimizer=adamw fused=false backend=torch_adamw device={device.type} dtype={dtype}")
            return torch.optim.AdamW(
                params,  # type: ignore[arg-type]
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=float(weight_decay),
            )

        # opt_name == "adam"
        logger.info(f"optimizer=adam backend=torch_adam device={device.type} dtype={dtype}")
        return torch.optim.Adam(
            params,  # type: ignore[arg-type]
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=float(weight_decay),
        )

    if opt_name == "sgd":
        return torch.optim.SGD(
            params,  # type: ignore[arg-type]
            lr=lr,
            weight_decay=float(weight_decay),
        )

    if opt_name == "lion":
        return Lion(
            params,  # type: ignore[arg-type]
            lr=lr,
            weight_decay=float(weight_decay),
            fused=bool(fused_opt),
        )

    raise ValueError(f"Unknown optimizer {opt_name!r}")

