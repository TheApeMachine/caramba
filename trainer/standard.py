"""Standard training loop for generic ML experiments.

Unlike upcycling (which uses a teacher/student setup), the Standard trainer
performs regular end-to-end training on a single model. This is the foundation
for training new architectures from scratch.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Sized

from torch.utils.data import DataLoader, Subset

from carmath import token_budget_batch_size
from config.defaults import Defaults
from config.group import Group
from config.manifest import Manifest
from config.run import Run
from config.train import TrainConfig
from console import logger
from data import build_token_dataset
from carmath import train_val_counts
from instrumentation import (
    LivePlotter,
    RunLogger,
    TensorBoardWriter,
    WandBWriter,
    generate_analysis_png,
)
from carmath import autocast_dtype, autocast_dtype_str, weight_dtype_str
from model import Model
from runtime import RuntimePlan, make_plan_key, save_plan, load_plan
from trainer.distributed import (
    DistributedConfig,
    DistributedContext,
    DistributedStrategy,
)
from trainer.scheduler import LRSchedulerConfig, build_lr_scheduler


class StandardTrainer:
    """Orchestrates standard end-to-end model training.

    Handles data loading, optimization, logging, and checkpointing for
    a single model architecture.
    """

    def __init__(
        self,
        manifest: Manifest,
        group: Group,
        train: TrainConfig,
        *,
        dist_config: DistributedConfig | None = None,
        defaults: Defaults | None = None,
        checkpoint_dir: Path | str | None = None,
        resume_from: Path | str | None = None,
    ) -> None:
        """Initialize the trainer and build the model.

        Args:
            manifest: Model architecture specification
            group: Experiment group with data paths and settings
            train: Training hyperparameters
            dist_config: Optional distributed training settings
            defaults: Optional global defaults
            checkpoint_dir: Where to save checkpoints
            resume_from: Path to resume training from a checkpoint
        """
        self.manifest = manifest
        self.group = group
        self.defaults = defaults
        self.train_cfg = train

        self.save_every = defaults.save_every if defaults else 500
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else Path("runs") / group.name
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_logger = RunLogger(self.checkpoint_dir, filename="train.jsonl", enabled=True)

        # Set up device and distribution
        self.dist_config = dist_config
        self.dist_ctx: DistributedContext | None = None
        if dist_config is not None and dist_config.strategy != DistributedStrategy.NONE:
            self.dist_ctx = DistributedContext.init(dist_config)
            self.device = self.dist_ctx.device
        else:
            self.device = torch.device(train.device)

        self.runtime_plan = self._load_or_create_runtime_plan(train)
        self.dtype = self._parse_dtype(self.runtime_plan.dtype)

        # Build model
        logger.info(f"Building model: {self.manifest.model.type}")
        self.model = Model(self.manifest.model).to(device=self.device, dtype=self.dtype)

        if self.dist_ctx is not None:
            self.model = self.dist_ctx.wrap_model(self.model)

        if bool(self.runtime_plan.compile) and hasattr(torch, "compile"):
            if self.device.type == "cuda":
                logger.info(f"Compiling model (mode={self.runtime_plan.compile_mode})...")
                self.model = torch.compile(self.model, mode=self.runtime_plan.compile_mode)
            else:
                logger.warning(f"torch.compile not supported on {self.device.type}")

        if resume_from is not None:
            self.load_checkpoint(Path(resume_from))

    def run(self, run: Run) -> None:
        """Execute the training run."""
        torch.manual_seed(run.seed)

        # Build data loaders
        loader, val_loader = self.build_loaders(self.train_cfg)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_cfg.lr)

        scheduler = build_lr_scheduler(
            optimizer,
            LRSchedulerConfig(
                kind=self.train_cfg.scheduler,
                total_steps=run.steps,
                warmup_steps=self.train_cfg.warmup_steps,
                min_lr_ratio=self.train_cfg.min_lr_ratio,
            ),
        )

        logger.header("Training", f"{run.steps} steps")
        self.model.train()

        loader_iter = iter(loader)
        use_amp = bool(self.train_cfg.use_amp)
        amp_dtype = autocast_dtype(self.device, str(self.train_cfg.amp_dtype))

        with logger.progress_bar() as progress:
            task = progress.add_task("Training...", total=run.steps)

            for step in range(run.steps):
                (x, y), loader_iter = self._next_batch(loader, loader_iter)
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    # Basic next-token prediction loss
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

                # Logging
                if (step + 1) % 10 == 0:
                    self.run_logger.log_metrics(
                        run_id=run.id,
                        phase="standard",
                        step=step + 1,
                        metrics={"loss": float(loss), "lr": optimizer.param_groups[0]["lr"]},
                    )

                progress.update(task, advance=1, description=f"Step {step+1}/{run.steps} • loss={float(loss):.4f}")

                if self.save_every > 0 and (step + 1) % self.save_every == 0:
                    self.save_checkpoint(run.id, "standard", step + 1)

        logger.success(f"Training complete • final loss={float(loss):.4f}")
        self.save_checkpoint(run.id, "standard", run.steps, is_final=True)

    def build_loaders(self, train: TrainConfig) -> tuple[DataLoader, DataLoader | None]:
        """Build train and validation data loaders."""
        path = Path(self.group.data)
        dataset = build_token_dataset(path=path, block_size=train.block_size)

        val_frac = self.defaults.val_frac if self.defaults else 0.0
        n = len(cast(Sized, dataset))
        n_train, n_val = train_val_counts(n, float(val_frac))

        train_ds = Subset(dataset, range(0, n_train))
        val_ds = Subset(dataset, range(n_train, n)) if n_val > 0 else None

        loader_kwargs = {
            "batch_size": self.runtime_plan.batch_size,
            "num_workers": train.num_workers,
            "pin_memory": train.pin_memory and self.device.type == "cuda",
            "drop_last": True,
        }

        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs) if val_ds else None

        return train_loader, val_loader

    def _next_batch(self, loader: DataLoader, iterator: Any) -> tuple[Any, Any]:
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iter = iter(loader)
            return next(new_iter), new_iter

    def save_checkpoint(self, run_id: str, phase: str, step: int, is_final: bool = False) -> Path:
        filename = f"{run_id}_{phase}_{'final' if is_final else f'step{step}'}.pt"
        path = self.checkpoint_dir / filename
        state = {
            "model_state_dict": self.model.state_dict(),
            "run_id": run_id,
            "step": step,
        }
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")

    def _parse_dtype(self, dtype: str) -> torch.dtype:
        dt = dtype.lower()
        if dt == "float32": return torch.float32
        if dt == "float16": return torch.float16
        if dt == "bfloat16": return torch.bfloat16
        return torch.float32

    def _resolve_amp_dtype(self, amp_dtype: str) -> torch.dtype:
        return autocast_dtype(self.device, str(amp_dtype))

    def _load_or_create_runtime_plan(self, train: TrainConfig) -> RuntimePlan:
        """Derive a runtime plan with automatic dtype/batch/compile decisions.

        The plan is persisted to disk so subsequent runs with the same
        configuration reuse the same decisions.
        """
        # Build a stable payload that excludes volatile fields
        train_payload = train.model_dump()
        train_payload.pop("teacher_ckpt", None)
        payload: dict[str, Any] = {
            "device": str(self.device),
            "torch": torch.__version__,
            "model": self.manifest.model.model_dump(),
            "train": train_payload,
        }
        key = make_plan_key(payload)
        plan_path = self.checkpoint_dir / "plans" / f"{key}.json"

        # Check for existing plan
        existing = load_plan(plan_path)
        if existing is not None and existing.key == key:
            logger.info(f"Reusing cached runtime plan: {key}")
            return existing

        # Resolve dtype: auto-detect best precision for device
        dtype_str = str(train.dtype).lower()
        if dtype_str == "auto":
            dtype_str = weight_dtype_str(self.device)

        # Resolve AMP dtype
        amp_dtype_str = str(train.amp_dtype).lower()
        if amp_dtype_str == "auto":
            amp_dtype_str = autocast_dtype_str(self.device)

        # Batch size tuning: scale based on block size ratio
        batch_size = int(train.batch_size)
        if bool(getattr(train, "auto_batch_size", False)):
            ref_block = int(getattr(train, "auto_batch_ref_block_size", 512))
            min_bs = int(getattr(train, "auto_batch_min", 1))
            batch_size = token_budget_batch_size(
                batch_size,
                block_size=int(train.block_size),
                ref_block_size=int(ref_block),
                min_batch_size=int(min_bs),
            )

            # Memory-aware batch size search
            batch_size = self._find_max_batch_size(train, batch_size, dtype_str)

        # Compile decision
        compile_setting = getattr(train, "compile_model", False)
        compile_mode = str(getattr(train, "compile_mode", "reduce-overhead"))
        should_compile = self._resolve_compile_setting(compile_setting)

        plan = RuntimePlan(
            key=key,
            device=str(self.device),
            torch_version=torch.__version__,
            dtype=dtype_str,
            use_amp=bool(train.use_amp),
            amp_dtype=amp_dtype_str,
            batch_size=batch_size,
            compile=should_compile,
            compile_mode=compile_mode,
        )

        # Persist plan
        try:
            save_plan(plan_path, plan, payload=payload)
            logger.info(f"Created runtime plan: {key}")
        except Exception as e:
            logger.warning(f"Failed to save runtime plan: {e}")

        return plan

    # dtype/amp auto selection moved into carmath.precision

    def _resolve_compile_setting(self, compile_setting: object) -> bool:
        """Resolve compile setting to boolean."""
        if isinstance(compile_setting, bool):
            return compile_setting
        s = str(compile_setting).strip().lower()
        if s == "auto":
            return self.device.type == "cuda"
        return s in ("1", "true", "yes", "on")

    def _find_max_batch_size(self, train: TrainConfig, initial_bs: int, dtype_str: str) -> int:
        """Binary search for maximum batch size that fits in memory.

        Performs a quick forward pass with increasing batch sizes to find
        the largest that doesn't OOM.
        """
        if self.device.type != "cuda":
            return initial_bs

        # Get available GPU memory
        try:
            props = torch.cuda.get_device_properties(self.device)
            total_mem = props.total_memory
            reserved_mem = torch.cuda.memory_reserved(self.device)
            available = total_mem - reserved_mem
        except Exception:
            return initial_bs

        # Estimate bytes per sample based on model size and sequence length
        # This is a heuristic based on typical transformer memory usage
        try:
            model_params = sum(p.numel() for p in self.model.parameters())
        except Exception:
            return initial_bs

        bytes_per_param = 2 if dtype_str in ("float16", "bfloat16") else 4

        # Rough estimate: activations + gradients ≈ 4x model size per sample
        # This varies by architecture but provides a reasonable starting point
        seq_len = train.block_size
        hidden_dim = getattr(self.manifest.model, "d_model", 512)
        activation_mem_per_sample = seq_len * hidden_dim * bytes_per_param * 4

        # Leave 20% headroom for fragmentation and other allocations
        usable_mem = available * 0.8

        # Calculate max batch size
        max_bs = max(1, int(usable_mem / activation_mem_per_sample))

        # Bound by initial and do binary search if there's significant headroom
        if max_bs <= initial_bs:
            return initial_bs

        # Binary search between initial and estimated max
        low, high = initial_bs, min(max_bs, initial_bs * 8)
        best_bs = initial_bs

        # Quick validation: try to allocate test tensors
        test_input_shape = (1, seq_len)

        while low <= high:
            mid = (low + high) // 2
            try:
                # Try allocating tensors of this batch size
                test_tensor = torch.empty(
                    mid, seq_len, hidden_dim,
                    device=self.device,
                    dtype=self._parse_dtype(dtype_str),
                )
                del test_tensor
                torch.cuda.empty_cache()
                best_bs = mid
                low = mid + 1
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                high = mid - 1

        logger.info(f"Auto batch size: {initial_bs} -> {best_bs}")
        return best_bs
