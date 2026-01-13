"""Standard training metrics.

This module holds the default step metrics computation used by the standard
training loop.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from caramba.carmath import global_grad_norm_l2, safe_perplexity_from_nll
from caramba.config.train import TrainConfig
from caramba.console import logger
from caramba.instrumentation.viz import TrainingVizMosaicContext
from caramba.runtime.plan import RuntimePlan
from caramba.runtime.tensordict_utils import TensorDictBase
from caramba.trainer.mosaic_table2 import Table2Telemetry


def _bytes_to_mb(n: int) -> float:
    return float(n) / (1024.0 * 1024.0)


def _grad_bytes(system: object) -> int:
    total = 0
    for p in system.parameters():  # type: ignore[attr-defined]
        g = getattr(p, "grad", None)
        if isinstance(g, Tensor):
            total += int(g.numel()) * int(g.element_size())
    return total


def _optim_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for _param, state in optimizer.state.items():
        if isinstance(state, dict):
            for v in state.values():
                if isinstance(v, Tensor):
                    total += int(v.numel()) * int(v.element_size())
    return total


def build_standard_step_metrics(
    *,
    system: object,
    train: TrainConfig,
    runtime_plan: RuntimePlan,
    optimizer: torch.optim.Optimizer,
    compiled: bool,
    step_time_s: float,
    data_time_s: float,
    fwd_bwd_time_s: float,
    optim_time_s: float,
    accum_steps: int,
    loss_sum: Tensor,
    loss_last: Tensor | None,
    outputs_last: object | None,
    last_batch_td: TensorDictBase | None,
    call_objective_metrics: Any,
    table2: Table2Telemetry,
    viz_ctx: TrainingVizMosaicContext,
    param_mb: float,
    kernel_events_estimate: int | None,
) -> dict[str, float]:
    # Loss value (mean over microbatches).
    loss_val = float((loss_sum / float(accum_steps)).item())
    ppl = float(safe_perplexity_from_nll(float(loss_val)))

    lr = float(optimizer.param_groups[0].get("lr", float(train.lr)))
    lr_base = float(getattr(train, "lr", lr))
    lr_mult = (lr / lr_base) if lr_base > 0 else 1.0

    # Grad norm.
    try:
        grad_norm = float(global_grad_norm_l2(system))  # type: ignore[arg-type]
    except Exception as e:
        raise RuntimeError("Failed to compute gradient norm") from e

    metrics: dict[str, float] = {
        "loss": float(loss_val),
        "train_loss": float(loss_val),
        "ppl": float(ppl),
        "train_ppl": float(ppl),
        "lr": float(lr),
        "lr_base": float(lr_base),
        "lr_mult": float(lr_mult),
        "grad_norm": float(grad_norm),
        "grad_accum": float(accum_steps),
        # Log the effective batch size (runtime plan can override train.batch_size).
        "batch_size": float(getattr(runtime_plan, "batch_size", getattr(train, "batch_size", 0))),
        "seq_len": float(getattr(train, "block_size", 0)),
        "compiled": 1.0 if compiled else 0.0,
        "time_step_s": float(step_time_s),
        "time_data_s": float(data_time_s),
        "time_fwd_bwd_s": float(fwd_bwd_time_s),
        "time_optim_s": float(optim_time_s),
        # Timing metrics (ms).
        "ms_step": float(step_time_s * 1000.0),
        "ms_data": float(data_time_s * 1000.0),
        "ms_fwd_bwd": float(fwd_bwd_time_s * 1000.0),
        "ms_opt": float(optim_time_s * 1000.0),
    }

    # Objective extra metrics.
    if outputs_last is not None and loss_last is not None and last_batch_td is not None:
        try:
            extra = call_objective_metrics(outputs=outputs_last, batch_td=last_batch_td, loss=loss_last)
            if isinstance(extra, dict):
                metrics.update({str(k): float(v) for k, v in extra.items()})
        except Exception as e:
            raise RuntimeError("Failed to compute objective metrics") from e

    # Table 2 telemetry.
    if last_batch_td is not None:
        try:
            has_table2_bin = isinstance(last_batch_td.get("table2_bin", None), Tensor)  # type: ignore[attr-defined]
            has_mem_teacher = (
                isinstance(last_batch_td.get("memblock_teacher_read_bucket", None), Tensor)  # type: ignore[attr-defined]
                and isinstance(last_batch_td.get("memblock_teacher_write_bucket", None), Tensor)  # type: ignore[attr-defined]
                and isinstance(last_batch_td.get("memblock_teacher_write_gate", None), Tensor)  # type: ignore[attr-defined]
            )
            # Debug scalars: help diagnose "all -1" / "no reads" situations quickly in W&B.
            if has_table2_bin:
                tb2 = last_batch_td.get("table2_bin", None)  # type: ignore[attr-defined]
                if isinstance(tb2, Tensor):
                    valid = (tb2.detach() >= 0)
                    metrics["table2/valid_frac"] = float(valid.float().mean().item()) if tb2.numel() > 0 else 0.0
                    metrics["table2/valid_count"] = float(valid.float().sum().item())
            if has_mem_teacher:
                rb2 = last_batch_td.get("memblock_teacher_read_bucket", None)  # type: ignore[attr-defined]
                wb2 = last_batch_td.get("memblock_teacher_write_bucket", None)  # type: ignore[attr-defined]
                wg2 = last_batch_td.get("memblock_teacher_write_gate", None)  # type: ignore[attr-defined]
                if isinstance(rb2, Tensor) and rb2.numel() > 0:
                    metrics["mem/teacher_read_frac"] = float((rb2.detach() >= 0).float().mean().item())
                if isinstance(wb2, Tensor) and wb2.numel() > 0:
                    metrics["mem/teacher_write_bucket_frac"] = float((wb2.detach() >= 0).float().mean().item())
                if isinstance(wg2, Tensor) and wg2.numel() > 0:
                    wg2f = wg2.detach().float()
                    metrics["mem/teacher_write_gate_mean"] = float(wg2f.mean().item())
                    metrics["mem/teacher_write_gate_fire_frac"] = float((wg2f > 0.5).float().mean().item())
            if (has_table2_bin or has_mem_teacher) and (outputs_last is not None):
                metrics.update(table2.compute(outputs=outputs_last, batch=last_batch_td))
        except Exception as e:
            raise RuntimeError("Failed to compute Table 2 telemetry") from e

    # Token throughput (for token-LM style datasets).
    if last_batch_td is not None:
        try:
            y = last_batch_td.get("target_ids", None)  # type: ignore[attr-defined]
            if isinstance(y, Tensor):
                metrics["tok_s"] = float(y.numel() * int(accum_steps)) / float(max(1e-9, step_time_s))
        except Exception as e:
            raise RuntimeError("Failed to compute token throughput") from e

    # Memory footprint estimates (MiB).
    metrics["mem_params_mb"] = float(param_mb)
    try:
        metrics["mem_grads_mb"] = float(_bytes_to_mb(_grad_bytes(system)))
    except Exception as e:
        logger.warning(f"Failed to estimate grad memory; continuing. error={e!r}")
    try:
        metrics["mem_optim_mb"] = float(_bytes_to_mb(_optim_state_bytes(optimizer)))
    except Exception as e:
        logger.warning(f"Failed to estimate optimizer memory; continuing. error={e!r}")

    if kernel_events_estimate is not None:
        metrics["kernel_events_estimate"] = float(kernel_events_estimate)

    # MOSAIC memory stats.
    try:
        if isinstance(viz_ctx, TrainingVizMosaicContext) and viz_ctx.memblock_mem_stats:
            mosaic_f: dict[str, float] = {}
            t_keys: list[str] = []
            t_vals: list[Tensor] = []
            for k, v in viz_ctx.memblock_mem_stats.items():
                kk = str(k)
                if isinstance(v, (int, float)):
                    mosaic_f[kk] = float(v)
                elif isinstance(v, Tensor) and v.numel() == 1:
                    t_keys.append(kk)
                    t_vals.append(v.detach().float())

            if t_vals:
                stacked = torch.stack(t_vals)
                flat = stacked.detach().cpu().tolist()
                for kk, vv in zip(t_keys, flat, strict=True):
                    mosaic_f[kk] = float(vv)

            for kk, vv in mosaic_f.items():
                metrics[kk] = float(vv)

            metrics["memblock_teacher_p"] = float(getattr(viz_ctx, "memblock_teacher_p", 1.0))

            def _avg(suffix: str) -> float | None:
                vals = [float(v) for kk, v in mosaic_f.items() if str(kk).endswith(suffix)]
                if not vals:
                    return None
                return float(sum(vals) / float(len(vals)))

            # Table 2 stable namespaces.
            rg = _avg("/fuse_gate_mem_mean")
            wg = _avg("/write_gate_p_mean")
            re = _avg("/write_bucket_entropy_norm")
            if rg is not None:
                metrics["mem/read_gate"] = float(rg)
            if wg is not None:
                metrics["mem/write_gate"] = float(wg)
            if re is not None:
                metrics["mem/routing_entropy"] = float(re)
    except Exception as e:
        raise RuntimeError("Failed to log MOSAIC memory stats") from e

    return metrics

