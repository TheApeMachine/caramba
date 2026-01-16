"""MLX Trainer for the Routing Hypothesis Experiment.

This trainer implements gradient isolation for testing the routing hypothesis:
- Freeze FFN layers and embeddings (they have the "knowledge")
- Train only fresh DBA attention layers
- Use standard language modeling loss

Performance optimizations (based on mlx-lm patterns):
- mx.compile on the step function with state capture
- tree_map for gradient operations (no Python loops)
- Proper lazy evaluation with batched mx.eval
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np

from caramba.adapter.mlx.surgery import AttentionSurgeryMLX
from caramba.console import logger
from caramba.layer.mlx.transformer import DBATransformer


@dataclass
class TrainConfig:
    """Training configuration."""

    # Data
    data_path: str
    block_size: int = 2048
    batch_size: int = 1

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 5000
    grad_accum_steps: int = 16
    grad_clip_norm: float = 1.0

    # Logging
    log_interval: int = 10
    save_interval: int = 500
    save_dir: str = "checkpoints"

    # LR schedule
    min_lr_ratio: float = 0.1


class TokenDataLoader:
    """Simple token data loader for MLX."""

    def __init__(
        self,
        data_path: str | Path,
        block_size: int,
        batch_size: int,
    ):
        # Load tokenized data (numpy memmap for large files)
        self.data = np.load(data_path, mmap_mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_tokens = len(self.data)

    def __iter__(self) -> Iterator[tuple[mx.array, mx.array]]:
        while True:
            # Random starting positions
            starts = np.random.randint(
                0, self.n_tokens - self.block_size - 1, size=self.batch_size
            )

            # Build batch
            batch = np.stack([
                self.data[s : s + self.block_size + 1]
                for s in starts
            ])

            # Split into input and target
            input_ids = mx.array(batch[:, :-1])
            targets = mx.array(batch[:, 1:])

            yield input_ids, targets


def count_params(params: Any) -> int:
    """Count total parameters in a nested dict/list structure."""
    if isinstance(params, mx.array):
        return params.size
    elif isinstance(params, dict):
        return sum(count_params(v) for v in params.values())
    elif isinstance(params, list):
        return sum(count_params(v) for v in params)
    return 0


def cosine_schedule(step: int, warmup_steps: int, max_steps: int, min_ratio: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return step / warmup_steps

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))


def build_gradient_mask(model: DBATransformer) -> dict:
    """Build a mask dict with 1.0 for attention params, 0.0 for others.

    This is computed once and used with tree_map for efficient masking.
    """
    def _build_mask(params: Any, prefix: str = "") -> Any:
        if isinstance(params, dict):
            return {k: _build_mask(v, f"{prefix}.{k}" if prefix else k) for k, v in params.items()}
        elif isinstance(params, list):
            return [_build_mask(v, f"{prefix}.{i}") for i, v in enumerate(params)]
        elif isinstance(params, mx.array):
            # Keep attention params, zero out others
            if "attention" in prefix or "gate" in prefix:
                return mx.ones_like(params)
            else:
                return mx.zeros_like(params)
        return params

    return _build_mask(model.parameters())


class RoutingHypothesisTrainer:
    """Trainer for the routing hypothesis experiment.

    Only trains attention parameters while keeping FFN/embeddings frozen.

    Uses mlx-lm style optimizations:
    - Compiled step function
    - tree_map for gradient operations
    - Proper state capture for mx.compile
    """

    def __init__(
        self,
        model: DBATransformer,
        config: TrainConfig,
        surgery: AttentionSurgeryMLX,
    ):
        self.model = model
        self.config = config
        self.surgery = surgery

        # Get only trainable (attention) parameters
        self.trainable_params = surgery.get_trainable_params(model)

        # Build gradient mask ONCE (1.0 for attention, 0.0 for frozen)
        self._grad_mask = build_gradient_mask(model)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            learning_rate=config.lr,
            weight_decay=config.weight_decay,
        )

        # Data loader
        self.data_loader = TokenDataLoader(
            config.data_path,
            config.block_size,
            config.batch_size,
        )

        # State for mx.compile (must include all mutable state)
        self._state = [model.state, self.optimizer.state]

        # Create loss function
        def loss_fn(model: DBATransformer, input_ids: mx.array, targets: mx.array) -> tuple[mx.array, int]:
            logits, _, _ = model(input_ids)  # Returns (logits, cache, attn_weights)
            B, T, V = logits.shape
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, V),
                targets.reshape(-1),
                reduction="mean"
            )
            return loss, B * T

        # Create value_and_grad function
        self._loss_value_and_grad = nn.value_and_grad(model, loss_fn)

        # Build compiled functions (separate for grad accumulation and update)
        self._grad_fn, self._update_fn = self._build_compiled_fns()

        # Training state
        self.step = 0
        self.loss_history: list[float] = []
        self.best_loss = float("inf")

    def _build_compiled_fns(self):
        """Build compiled training functions following mlx-lm patterns.

        We create two separate compiled functions:
        1. grad_fn: Computes loss and masked gradients (called every micro-batch)
        2. update_fn: Averages gradients and updates model (called every grad_accum_steps)

        This avoids Python control flow inside compiled functions.
        """
        model = self.model
        optimizer = self.optimizer
        grad_mask = self._grad_mask
        loss_value_and_grad = self._loss_value_and_grad
        state = self._state

        # Compiled gradient function - computes one forward/backward pass
        @partial(mx.compile, inputs=state, outputs=state)
        def grad_fn(input_ids: mx.array, targets: mx.array):
            (loss, ntoks), grads = loss_value_and_grad(model, input_ids, targets)
            # Mask gradients (zero out non-attention params) using tree_map
            grads = tree_map(lambda g, m: g * m, grads, grad_mask)
            return loss, ntoks, grads

        # Compiled update function - scales gradients, clips, and updates model
        grad_clip_norm = self.config.grad_clip_norm

        @partial(mx.compile, inputs=state, outputs=state)
        def update_fn(grads: Any, scale: mx.array):
            # Scale gradients (divide by accumulation steps)
            scaled_grads = tree_map(lambda g: g * scale, grads)

            # Gradient clipping by global norm
            if grad_clip_norm > 0:
                # Compute global norm
                flat_grads = tree_flatten(scaled_grads)
                total_norm_sq = sum(mx.sum(g * g) for _, g in flat_grads)
                total_norm = mx.sqrt(total_norm_sq)

                # Clip factor: min(1.0, clip_norm / total_norm)
                clip_coef = mx.minimum(mx.array(1.0), mx.array(grad_clip_norm) / (total_norm + 1e-6))
                scaled_grads = tree_map(lambda g: g * clip_coef, scaled_grads)

            # Update model
            optimizer.update(model, scaled_grads)

        return grad_fn, update_fn

    def train(self) -> None:
        """Main training loop with rich progress tracking."""
        logger.header("MLX Routing Hypothesis Training", f"{self.config.max_steps} steps")

        # Display config
        logger.key_value({
            "Batch size": self.config.batch_size,
            "Block size": self.config.block_size,
            "Grad accum": self.config.grad_accum_steps,
            "Effective batch": self.config.batch_size * self.config.grad_accum_steps,
            "Learning rate": f"{self.config.lr:.2e}",
            "Warmup steps": self.config.warmup_steps,
            "Trainable groups": len(self.trainable_params),
        }, title="Training Config")

        data_iter = iter(self.data_loader)
        start_time = time.time()

        # Precompute gradient scale (1/grad_accum_steps)
        grad_scale = mx.array(1.0 / self.config.grad_accum_steps)

        # Use rich progress bar
        with logger.progress_bar() as progress:
            task = progress.add_task(
                "[info]Training attention layers...[/info]",
                total=self.config.max_steps,
            )

            while self.step < self.config.max_steps:
                # Update learning rate
                lr_scale = cosine_schedule(
                    self.step,
                    self.config.warmup_steps,
                    self.config.max_steps,
                    self.config.min_lr_ratio,
                )
                self.optimizer.learning_rate = self.config.lr * lr_scale

                # Accumulate gradients over micro-batches
                # IMPORTANT: Evaluate after each micro-batch to avoid OOM
                # MLX lazy eval would otherwise hold all activations in memory
                accum_grads = None
                accum_loss = mx.array(0.0)

                for _ in range(self.config.grad_accum_steps):
                    # Get batch
                    input_ids, targets = next(data_iter)

                    # Compiled gradient step
                    loss, ntoks, grads = self._grad_fn(input_ids, targets)

                    # Accumulate gradients
                    if accum_grads is None:
                        accum_grads = grads
                    else:
                        accum_grads = tree_map(lambda a, g: a + g, accum_grads, grads)
                    accum_loss = accum_loss + loss

                    # CRITICAL: Evaluate after each micro-batch to free memory
                    # This materializes the gradient accumulation without holding
                    # all forward pass activations in memory
                    mx.eval(accum_grads, accum_loss)

                # Average loss (for logging)
                avg_loss = accum_loss / self.config.grad_accum_steps

                # Compiled update step (scales gradients internally)
                self._update_fn(accum_grads, grad_scale)

                # Force evaluation after optimizer step
                mx.eval(self._state, avg_loss)

                loss_val = float(avg_loss)
                self.step += 1

                # Track metrics
                self.loss_history.append(loss_val)
                if loss_val < self.best_loss:
                    self.best_loss = loss_val

                # Update progress bar
                elapsed = time.time() - start_time
                tok_per_sec = (
                    self.step
                    * self.config.batch_size
                    * self.config.block_size
                    * self.config.grad_accum_steps
                    / elapsed
                )

                # Compute perplexity
                ppl = math.exp(loss_val) if loss_val < 20 else float("inf")

                # Update progress with metrics
                progress.update(
                    task,
                    completed=self.step,
                    description=(
                        f"[info]step {self.step}/{self.config.max_steps}[/info] "
                        f"[metric]loss={loss_val:.4f}[/metric] "
                        f"[muted]ppl={ppl:.1f}[/muted] "
                        f"[step]lr={self.optimizer.learning_rate:.2e}[/step] "
                        f"[muted]{tok_per_sec:.0f} tok/s[/muted]"
                    ),
                )

                # Save checkpoint
                if self.step > 0 and self.step % self.config.save_interval == 0:
                    self.save_checkpoint()

        # Final summary
        logger.success(f"Training complete! Final loss: {self.loss_history[-1]:.4f}")
        logger.key_value({
            "Final loss": f"{self.loss_history[-1]:.4f}",
            "Best loss": f"{self.best_loss:.4f}",
            "Final PPL": f"{math.exp(self.loss_history[-1]):.2f}",
            "Total time": f"{time.time() - start_time:.1f}s",
        }, title="Training Summary")

        self.save_checkpoint()

    def save_checkpoint(self) -> None:
        """Save model checkpoint."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Update trainable params from model
        self.trainable_params = self.surgery.get_trainable_params(self.model)

        path = save_dir / f"checkpoint_{self.step}.npz"
        # Save only trainable params (attention)
        mx.savez(str(path), **self.trainable_params)
        logger.success(f"Checkpoint saved: {path}")


def run_routing_hypothesis_mlx(
    teacher_weights_path: str,
    data_path: str,
    *,
    sem_dim: int = 256,
    geo_dim: int = 512,
    v_dim: int = 768,
    max_steps: int = 5000,
    lr: float = 1e-4,
    **kwargs: Any,
) -> None:
    """Run the routing hypothesis experiment with MLX.

    This is the main entry point for testing whether attention is primarily
    a routing mechanism.

    Args:
        teacher_weights_path: Path to pretrained Llama weights
        data_path: Path to tokenized training data (.npy)
        sem_dim: Semantic dimension for DBA
        geo_dim: Geometric dimension for DBA
        v_dim: Value dimension for DBA
        max_steps: Number of training steps
        lr: Learning rate
        **kwargs: Additional config options
    """
    logger.header("MLX Routing Hypothesis Experiment")

    # Create surgery adapter
    surgery = AttentionSurgeryMLX(
        sem_dim=sem_dim,
        geo_dim=geo_dim,
        v_dim=v_dim,
    )

    # Load weights and create model
    logger.info(f"Loading teacher weights from {teacher_weights_path}")
    teacher_weights = surgery.load_llama_weights(teacher_weights_path)

    logger.info("Creating DBA model...")
    model = surgery.create_dba_model()

    logger.info("Applying attention surgery (fresh init)...")
    model = surgery.apply_surgery(model, teacher_weights, init_mode="fresh")

    # Count params
    total_params = count_params(model.parameters())
    trainable_params = count_params(surgery.get_trainable_params(model))

    logger.key_value({
        "Total params": f"{total_params:,}",
        "Trainable params": f"{trainable_params:,}",
        "Trainable %": f"{100*trainable_params/total_params:.1f}%",
        "Semantic dim": sem_dim,
        "Geometric dim": geo_dim,
        "Value dim": v_dim,
    }, title="Model Config")

    # Create config
    config = TrainConfig(
        data_path=data_path,
        max_steps=max_steps,
        lr=lr,
        **kwargs,
    )

    # Train
    trainer = RoutingHypothesisTrainer(model, config, surgery)
    trainer.train()

    logger.header("Experiment Complete")
