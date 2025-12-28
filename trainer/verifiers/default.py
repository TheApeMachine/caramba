"""Default verifier implementation for Upcycle."""

from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from config.eval import EvalVerifyConfig
from config.verifier import DefaultVerifierConfig
from config.verify import CompareVerifyConfig, FidelityVerifyConfig, KVCacheVerifyConfig
from console import logger
from data import build_token_dataset
from eval.suite import assert_eval_thresholds, run_eval_verify
from layer.attention import AttentionLayer
from carmath import train_val_counts
from trainer.compare import assert_thresholds, compare_teacher_student
from trainer.fidelity import assert_fidelity_thresholds, compute_short_context_fidelity
from trainer.upcycle_context import UpcycleContext
from trainer.verifiers.kvcache import (
    estimate_model_kvcache_bytes,
    estimate_model_kvcache_bytes_decoupled,
)
from runtime.tensordict_utils import TensorDictBase, as_tensordict, collate_tensordict


class DefaultVerifier:
    """Runs the existing verify config union (compare/eval/fidelity/kvcache)."""

    def __init__(self, config: DefaultVerifierConfig) -> None:
        self.config = config

    def verify(self, run: object, ctx: UpcycleContext) -> None:
        if not bool(self.config.enabled):
            return

        cfg = getattr(run, "verify", None)
        if cfg is None:
            return

        logger.header("Verification")

        if isinstance(cfg, CompareVerifyConfig):
            self._verify_compare(run, cfg, ctx)
            return
        if isinstance(cfg, EvalVerifyConfig):
            self._verify_eval(run, cfg, ctx)
            return
        if isinstance(cfg, FidelityVerifyConfig):
            self._verify_fidelity(run, cfg, ctx)
            return
        if isinstance(cfg, KVCacheVerifyConfig):
            self._verify_kvcache(run, cfg, ctx)
            return

        raise TypeError(f"Unsupported verify config: {type(cfg).__name__}")

    def _verify_compare(self, run: object, cfg: CompareVerifyConfig, ctx: UpcycleContext) -> None:
        train = getattr(run, "train", None)
        if train is None:
            raise ValueError("Run has no train config.")

        batches = self._collect_compare_batches(
            group_data=ctx.group.data,
            device=ctx.device,
            batch_size=int(train.batch_size),
            block_size=int(train.block_size),
            count=int(cfg.batches),
        )

        result = compare_teacher_student(
            teacher=ctx.teacher,
            student=ctx.student,
            batches=batches,
            predicate=lambda _, m: isinstance(m, AttentionLayer),
            attention=cfg.attention,
            logits=cfg.logits,
        )
        logger.success(f"Comparison complete • {result.batches} batches verified")

        metrics: dict[str, str] = {}
        if result.attention_mean_l1 is not None:
            metrics["attention_mean_l1"] = f"{result.attention_mean_l1:.6f}"
        if result.attention_max_l1 is not None:
            metrics["attention_max_l1"] = f"{result.attention_max_l1:.6f}"
        if result.logits_mean_l1 is not None:
            metrics["logits_mean_l1"] = f"{result.logits_mean_l1:.6f}"
        if result.logits_max_l1 is not None:
            metrics["logits_max_l1"] = f"{result.logits_max_l1:.6f}"
        if metrics:
            logger.key_value(metrics)
            if ctx.inst:
                ctx.inst.log_scalars(
                    step=0,
                    prefix="verify/compare",
                    scalars={k: float(v) for k, v in metrics.items()},
                )

        violations = assert_thresholds(
            result=result,
            attention=cfg.attention,
            logits=cfg.logits,
            fail_fast=cfg.fail_fast,
        )
        if violations:
            for v in violations:
                logger.warning(f"Threshold exceeded: {v.message()}")
            logger.warning(
                f"Verification found {len(violations)} threshold violation(s), "
                "but fail_fast=False so continuing..."
            )

    def _verify_eval(self, _run: object, cfg: EvalVerifyConfig, ctx: UpcycleContext) -> None:
        logger.info("Running behavioral evaluation suite...")
        summary = run_eval_verify(
            teacher=ctx.teacher,
            student=ctx.student,
            cfg=cfg,
            device=ctx.device,
        )
        logger.key_value(
            {
                "Teacher accuracy": f"{summary.teacher_accuracy:.1%}",
                "Student accuracy": f"{summary.student_accuracy:.1%}",
            }
        )
        if ctx.inst:
            ctx.inst.log_scalars(
                step=0,
                prefix="verify/eval",
                scalars={
                    "teacher_accuracy": float(summary.teacher_accuracy),
                    "student_accuracy": float(summary.student_accuracy),
                },
            )
        logger.success("Evaluation complete")
        assert_eval_thresholds(summary=summary, thresholds=cfg.thresholds)

    def _verify_fidelity(self, run: object, cfg: FidelityVerifyConfig, ctx: UpcycleContext) -> None:
        train = getattr(run, "train", None)
        if train is None:
            raise ValueError("Run has no train config.")

        batch_size = int(cfg.batch_size) if cfg.batch_size is not None else int(train.batch_size)
        block_size = int(cfg.block_size) if cfg.block_size is not None else int(train.block_size)

        logger.info(
            f"Running short-context fidelity on {cfg.batches} batches "
            f"(split={cfg.split}, batch_size={batch_size}, block_size={block_size})..."
        )

        batches = self._collect_fidelity_batches(
            group_data=ctx.group.data,
            device=ctx.device,
            defaults=ctx.defaults,
            batch_size=batch_size,
            block_size=block_size,
            count=int(cfg.batches),
            split=str(cfg.split),
        )

        result = compute_short_context_fidelity(
            teacher=ctx.teacher,
            student=ctx.student,
            batches=batches,
        )

        metrics = {
            "teacher_nll": float(result.teacher_nll),
            "student_nll": float(result.student_nll),
            "delta_nll": float(result.delta_nll),
            "ppl_ratio": float(result.ppl_ratio),
            "tokens": float(result.tokens),
        }
        logger.key_value(
            {
                "teacher_nll": f"{result.teacher_nll:.6f}",
                "student_nll": f"{result.student_nll:.6f}",
                "delta_nll": f"{result.delta_nll:.6f}",
                "ppl_ratio": f"{result.ppl_ratio:.6f}",
                "tokens": str(result.tokens),
            }
        )
        if ctx.inst:
            ctx.inst.log_scalars(step=0, prefix="verify/fidelity", scalars=metrics)

        violations = assert_fidelity_thresholds(
            result=result,
            max_delta_nll=float(cfg.max_delta_nll) if cfg.max_delta_nll is not None else None,
            max_ppl_ratio=float(cfg.max_ppl_ratio) if cfg.max_ppl_ratio is not None else None,
            fail_fast=bool(cfg.fail_fast),
        )
        if violations:
            for v in violations:
                logger.warning(f"Threshold exceeded: {v.message()}")
            logger.warning(
                f"Verification found {len(violations)} threshold violation(s), "
                "but fail_fast=False so continuing..."
            )

    def _verify_kvcache(self, _run: object, cfg: KVCacheVerifyConfig, ctx: UpcycleContext) -> None:
        logger.info("Analyzing KV-cache memory footprint...")

        teacher_bytes = estimate_model_kvcache_bytes(ctx.teacher, cfg.teacher, cfg.n_layers)
        student_bytes = estimate_model_kvcache_bytes_decoupled(ctx.student, cfg.student, cfg.n_layers)

        teacher_total = teacher_bytes * cfg.batch_size * cfg.max_seq_len
        student_total = student_bytes * cfg.batch_size * cfg.max_seq_len
        reduction = teacher_total / student_total if student_total > 0 else float("inf")

        logger.key_value(
            {
                "Teacher KV-cache": f"{teacher_total / 1024 / 1024:.2f} MB",
                "Student KV-cache": f"{student_total / 1024 / 1024:.2f} MB",
                "Reduction": f"{reduction:.2f}x",
            }
        )
        if ctx.inst:
            ctx.inst.log_scalars(
                step=0,
                prefix="verify/kvcache",
                scalars={
                    "teacher_kvcache_mb": float(teacher_total / 1024 / 1024),
                    "student_kvcache_mb": float(student_total / 1024 / 1024),
                    "reduction": float(reduction),
                },
            )
        logger.success("KV-cache analysis complete")

        if cfg.min_reduction_ratio is not None and reduction < cfg.min_reduction_ratio:
            raise AssertionError(
                f"KV cache reduction {reduction:.2f}x is below minimum {cfg.min_reduction_ratio}x"
            )

    @staticmethod
    def _collect_compare_batches(
        *,
        group_data: str,
        device: torch.device,
        batch_size: int,
        block_size: int,
        count: int,
    ) -> list[Tensor]:
        path = Path(group_data)
        dataset = build_token_dataset(path=path, block_size=int(block_size))
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=True,
            collate_fn=collate_tensordict,
        )

        batches: list[Tensor] = []
        for batch in loader:
            batches.append(batch["input_ids"].to(device=device))
            if len(batches) >= int(count):
                break
        return batches

    @staticmethod
    def _collect_fidelity_batches(
        *,
        group_data: str,
        device: torch.device,
        defaults: object | None,
        batch_size: int,
        block_size: int,
        count: int,
        split: str,
    ) -> list[TensorDictBase]:
        path = Path(group_data)
        dataset = build_token_dataset(path=path, block_size=int(block_size))

        val_frac = float(getattr(defaults, "val_frac", 0.0)) if defaults else 0.0
        n = len(cast(Sized, dataset))
        n_train, n_val = train_val_counts(n, val_frac)

        use_val = False
        if split in ("val", "auto") and n_val > 0 and n_train > 0:
            use_val = True
        if split == "val" and not use_val:
            logger.warning("Requested split=val, but no val split is configured; falling back to train.")

        if use_val:
            ds = Subset(dataset, range(n_train, n_train + n_val))
        elif n_val > 0 and n_train > 0:
            ds = Subset(dataset, range(0, n_train))
        else:
            ds = dataset

        loader = DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=False,
            drop_last=True,
            collate_fn=collate_tensordict,
        )
        batches: list[TensorDictBase] = []
        for batch in loader:
            td = as_tensordict(batch)
            td = td.to(device=device)
            batches.append(td)
            if len(batches) >= int(count):
                break
        return batches

