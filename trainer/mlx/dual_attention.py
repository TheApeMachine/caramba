"""Dual Attention Training: Learn the semantic/geometric decomposition.

This trainer runs BOTH original attention and DBA in parallel:
1. Original Llama attention (frozen) - provides the "ground truth" output
2. Fresh DBA attention (learning) - learns to decompose into sem/geo

The key insight: by training DBA to match the original attention output,
we can observe which input dimensions map to semantic vs geometric paths.
After training, we can extract projection matrices to convert original
attention weights into proper DBA weights.

The training signal is:
- Loss = MSE(DBA_output, Original_output) + small LM loss for stability

After training, we analyze:
- P_sem = Q_sem @ Original_Q.T  -> projection from original to semantic
- P_geo = Q_geo @ Original_Q.T  -> projection from original to geometric
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
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
from caramba.layer.mlx.standard_attention import TeacherModel


@dataclass
class DualAttentionConfig:
    """Configuration for dual attention training."""

    # Data
    data_path: str
    block_size: int = 2048
    batch_size: int = 1

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 5000
    grad_clip_norm: float = 1.0

    # Loss weights
    output_match_alpha: float = 1.0  # Weight for matching attention outputs
    lm_alpha: float = 0.01  # Small LM loss for stability

    # Logging
    log_interval: int = 10
    save_interval: int = 500
    save_dir: str = "checkpoints"

    # LR schedule
    min_lr_ratio: float = 0.1


class TokenDataLoader:
    """Simple token data loader for MLX."""

    def __init__(self, data_path: str | Path, block_size: int, batch_size: int):
        self.data = np.load(data_path, mmap_mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_tokens = len(self.data)

    def __iter__(self) -> Iterator[tuple[mx.array, mx.array]]:
        while True:
            starts = np.random.randint(
                0, self.n_tokens - self.block_size - 1, size=self.batch_size
            )
            batch = np.stack([
                self.data[s : s + self.block_size + 1]
                for s in starts
            ])
            input_ids = mx.array(batch[:, :-1])
            targets = mx.array(batch[:, 1:])
            yield input_ids, targets


def count_params(params: Any) -> int:
    """Count total parameters."""
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


class DualAttentionBlock(nn.Module):
    """A transformer block with BOTH original and DBA attention.

    Runs both attention mechanisms on the same input. The original attention
    is frozen; DBA learns to match its attention output.

    OPTIMIZATION: We only need teacher attention output for the matching loss,
    not the full teacher forward pass. The DBA path runs full forward (attention
    + FFN) because we need its logits for the LM loss.
    """

    def __init__(
        self,
        original_block,  # TeacherTransformerBlock
        dba_block,  # TransformerBlock with DBA attention
    ):
        super().__init__()
        self.original = original_block
        self.dba = dba_block

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Forward pass through both attention mechanisms.

        Returns:
            dba_out: Full output from DBA path (attention + FFN + residual)
            original_attn_out: Just the attention output from teacher (for matching)
            dba_attn_out: Just the DBA attention output (for matching)
        """
        # Original attention path (frozen) - ONLY compute attention, skip FFN
        h_orig = self.original.norm1(x)
        attn_orig, _, _ = self.original.attention(h_orig, mask=mask, return_weights=False)

        # DBA attention path (learning) - FULL forward pass
        h_dba = self.dba.norm1(x)
        attn_dba, _, _ = self.dba.attention(h_dba, mask=mask, return_weights=False)

        # DBA full block output (attention + FFN + residual)
        x_dba = x + attn_dba
        h_dba2 = self.dba.norm2(x_dba)
        ffn_dba = self.dba.ffn(h_dba2)
        out_dba = x_dba + ffn_dba

        return out_dba, attn_orig, attn_dba


class DualAttentionModel(nn.Module):
    """Model that runs both original and DBA attention in parallel."""

    def __init__(
        self,
        teacher: TeacherModel,
        student: DBATransformer,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student

        # Create dual blocks
        self.dual_blocks = []
        for i in range(len(teacher.layers)):
            dual = DualAttentionBlock(teacher.layers[i], student.layers[i])
            self.dual_blocks.append(dual)

        # Shared embeddings and head (from teacher)
        self.embed_tokens = teacher.embed_tokens
        self.norm = teacher.norm
        self.lm_head = teacher.lm_head

    def __call__(
        self,
        input_ids: mx.array,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        """Forward pass through dual attention model.

        OPTIMIZATION: Teacher only computes attention (not FFN) since we only
        need attention outputs for the matching loss. DBA runs full forward.

        Returns:
            dba_logits: Logits from DBA attention path (for LM loss)
            layer_outputs: List of (original_attn_out, dba_attn_out) per layer
        """
        x = self.embed_tokens(input_ids)

        layer_outputs = []
        x_dba = x

        for dual_block in self.dual_blocks:
            # Run teacher attention (only) and DBA full forward
            # Note: teacher attention gets same input as DBA each layer
            out_dba, attn_orig, attn_dba = dual_block(x_dba, mask=mask)
            layer_outputs.append((attn_orig, attn_dba))
            x_dba = out_dba

        # Compute DBA logits
        h_dba = self.student.norm(x_dba)
        logits_dba = self.student.lm_head(h_dba)

        return logits_dba, layer_outputs


class DualAttentionTrainer:
    """Trainer for dual attention learning."""

    def __init__(
        self,
        dual_model: DualAttentionModel,
        config: DualAttentionConfig,
        surgery: AttentionSurgeryMLX,
    ):
        self.model = dual_model
        self.config = config
        self.surgery = surgery

        # Build gradient mask (only DBA attention params)
        self._grad_mask = self._build_grad_mask()

        # Optimizer
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

        # State
        self._state = [dual_model.student.state, self.optimizer.state]
        self.step = 0
        self.loss_history: list[float] = []
        self.output_match_history: list[float] = []
        self.lm_loss_history: list[float] = []

    def _build_grad_mask(self) -> dict:
        """Build gradient mask: 1.0 for DBA attention, 0.0 for everything else."""
        def _mask(params: Any, prefix: str = "") -> Any:
            if isinstance(params, dict):
                return {k: _mask(v, f"{prefix}.{k}" if prefix else k) for k, v in params.items()}
            elif isinstance(params, list):
                return [_mask(v, f"{prefix}.{i}") for i, v in enumerate(params)]
            elif isinstance(params, mx.array):
                # Only train student (DBA) attention parameters
                if "student" in prefix and "attention" in prefix:
                    return mx.ones_like(params)
                return mx.zeros_like(params)
            return params
        return _mask(self.model.parameters())

    def train(self) -> None:
        """Main training loop."""
        logger.header("Dual Attention Training", f"{self.config.max_steps} steps")
        logger.info("Note: Running teacher attention + DBA full forward per step")

        logger.key_value({
            "Batch size": self.config.batch_size,
            "Block size": self.config.block_size,
            "Learning rate": f"{self.config.lr:.2e}",
            "Output match alpha": self.config.output_match_alpha,
            "LM alpha": self.config.lm_alpha,
        }, title="Training Config")

        data_iter = iter(self.data_loader)
        start_time = time.time()

        with logger.progress_bar() as progress:
            task = progress.add_task(
                "[info]Learning attention decomposition...[/info]",
                total=self.config.max_steps,
            )

            while self.step < self.config.max_steps:
                # Update LR
                lr_scale = cosine_schedule(
                    self.step,
                    self.config.warmup_steps,
                    self.config.max_steps,
                    self.config.min_lr_ratio,
                )
                self.optimizer.learning_rate = self.config.lr * lr_scale

                # Get batch
                input_ids, targets = next(data_iter)

                # Forward + backward
                loss, output_match, lm_loss, grads = self._forward_backward(
                    input_ids, targets
                )

                # Mask gradients
                grads = tree_map(lambda g, m: g * m, grads, self._grad_mask)

                # Gradient clipping
                if self.config.grad_clip_norm > 0:
                    flat_grads = tree_flatten(grads)
                    total_norm_sq = sum(mx.sum(g * g) for _, g in flat_grads)
                    total_norm = mx.sqrt(total_norm_sq)
                    clip_coef = mx.minimum(
                        mx.array(1.0),
                        mx.array(self.config.grad_clip_norm) / (total_norm + 1e-6)
                    )
                    grads = tree_map(lambda g: g * clip_coef, grads)

                # Update
                self.optimizer.update(self.model, grads)
                mx.eval(self._state, loss, output_match, lm_loss)

                loss_val = float(loss)
                output_match_val = float(output_match)
                lm_val = float(lm_loss)
                ppl_val = math.exp(min(lm_val, 20.0))  # Cap to avoid overflow
                self.step += 1

                self.loss_history.append(loss_val)
                self.output_match_history.append(output_match_val)
                self.lm_loss_history.append(lm_val)

                # Progress
                elapsed = time.time() - start_time
                tok_per_sec = self.step * self.config.batch_size * self.config.block_size / elapsed

                progress.update(
                    task,
                    completed=self.step,
                    description=(
                        f"[info]step {self.step}/{self.config.max_steps}[/info] "
                        f"[metric]loss={loss_val:.4f}[/metric] "
                        f"[muted]match={output_match_val:.4f} ppl={ppl_val:.1f}[/muted] "
                        f"[muted]{tok_per_sec:.0f} tok/s[/muted]"
                    ),
                )

                # Save checkpoint
                if self.step > 0 and self.step % self.config.save_interval == 0:
                    self.save_checkpoint()

        logger.success(f"Training complete! Final loss: {self.loss_history[-1]:.4f}")
        self.save_checkpoint()

        # Extract and save projection matrices
        self.extract_projections()

    def _forward_backward(
        self,
        input_ids: mx.array,
        targets: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, Any]:
        """Forward pass and compute gradients."""

        def loss_fn(model: DualAttentionModel):
            # Forward through dual model
            logits_dba, layer_outputs = model(input_ids)

            # Output matching loss: DBA attention output should match original
            output_match_loss = mx.array(0.0)
            for attn_orig, attn_dba in layer_outputs:
                diff = attn_orig - attn_dba
                layer_loss = mx.mean(diff * diff)
                output_match_loss = output_match_loss + layer_loss
            output_match_loss = output_match_loss / len(layer_outputs)

            # LM loss (for stability, using DBA logits)
            B, T, V = logits_dba.shape
            lm_loss = nn.losses.cross_entropy(
                logits_dba.reshape(-1, V),
                targets.reshape(-1),
                reduction="mean"
            )

            # Combined loss
            total_loss = (
                self.config.output_match_alpha * output_match_loss +
                self.config.lm_alpha * lm_loss
            )

            return total_loss, (output_match_loss, lm_loss)

        (total_loss, (output_match_loss, lm_loss)), grads = nn.value_and_grad(
            self.model, loss_fn
        )(self.model)

        return total_loss, output_match_loss, lm_loss, grads

    def save_checkpoint(self) -> None:
        """Save DBA attention checkpoint."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save all DBA attention params
        trainable = self.surgery.get_trainable_params(self.model.student, include_vo=True)
        path = save_dir / f"dual_checkpoint_{self.step}.npz"
        mx.savez(str(path), **trainable)
        logger.success(f"Checkpoint saved: {path}")

    def extract_projections(self) -> None:
        """Extract projection matrices from learned DBA weights.

        After training, DBA has learned Q_sem, K_sem, Q_geo, K_geo that decompose
        the original attention. We can analyze these to understand the decomposition
        and potentially transfer to new models.
        """
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        projections = {}

        for i in range(len(self.model.student.layers)):
            dba_attn = self.model.student.layers[i].attention
            orig_attn = self.model.teacher.layers[i].attention

            # Get the learned DBA projections
            q_sem = dba_attn.q_sem.weight  # (sem_dim, d_model)
            k_sem = dba_attn.k_sem.weight
            q_geo = dba_attn.q_geo.weight  # (geo_dim, d_model)
            k_geo = dba_attn.k_geo.weight

            # Get original Q/K
            q_orig = orig_attn.q_proj.weight  # (n_heads * head_dim, d_model)
            k_orig = orig_attn.k_proj.weight

            # Compute projection matrices: how to go from original to DBA
            # P_sem = Q_sem @ Q_orig.T @ (Q_orig @ Q_orig.T)^-1
            # Simplified: just store the correlation
            q_sem_corr = q_sem @ q_orig.T  # (sem_dim, n_heads * head_dim)
            q_geo_corr = q_geo @ q_orig.T  # (geo_dim, n_heads * head_dim)

            projections[f"layer_{i}_q_sem"] = q_sem
            projections[f"layer_{i}_k_sem"] = k_sem
            projections[f"layer_{i}_q_geo"] = q_geo
            projections[f"layer_{i}_k_geo"] = k_geo
            projections[f"layer_{i}_q_sem_corr"] = q_sem_corr
            projections[f"layer_{i}_q_geo_corr"] = q_geo_corr

        # Save projections
        path = save_dir / "learned_projections.npz"
        mx.savez(str(path), **projections)
        logger.success(f"Projections saved: {path}")

        # Log some statistics about the decomposition
        logger.header("Decomposition Analysis")
        for i in range(min(3, len(self.model.student.layers))):  # First 3 layers
            q_sem_corr = projections[f"layer_{i}_q_sem_corr"]
            q_geo_corr = projections[f"layer_{i}_q_geo_corr"]

            # Check how "clean" the decomposition is
            # High values in sem_corr and low in geo_corr (or vice versa) = clean split
            sem_energy = float(mx.mean(mx.abs(q_sem_corr)))
            geo_energy = float(mx.mean(mx.abs(q_geo_corr)))

            logger.info(f"Layer {i}: sem_energy={sem_energy:.4f}, geo_energy={geo_energy:.4f}")


def run_dual_attention_training(
    teacher_weights_path: str,
    data_path: str,
    *,
    sem_dim: int = 256,
    geo_dim: int = 512,
    v_dim: int = 768,
    max_steps: int = 5000,
    lr: float = 1e-4,
    output_match_alpha: float = 1.0,
    lm_alpha: float = 0.01,
    **kwargs: Any,
) -> None:
    """Run dual attention training experiment.

    This trains DBA to match the original attention output, learning
    the optimal semantic/geometric decomposition in the process.
    """
    from caramba.trainer.mlx.attention_distillation import load_teacher_from_llama

    logger.header("Dual Attention Training Experiment")

    # Load teacher model
    logger.info(f"Loading teacher from {teacher_weights_path}")
    teacher = load_teacher_from_llama(teacher_weights_path)
    teacher_params = count_params(teacher.parameters())
    logger.info(f"Teacher loaded: {teacher_params:,} params")

    # Create surgery adapter
    surgery = AttentionSurgeryMLX(
        sem_dim=sem_dim,
        geo_dim=geo_dim,
        v_dim=v_dim,
    )

    # Load teacher weights and create student
    teacher_weights = surgery.load_llama_weights(teacher_weights_path)

    logger.info("Creating DBA student model...")
    student = surgery.create_dba_model()

    # Apply surgery with fresh attention (we want to learn the decomposition)
    logger.info("Applying attention surgery (fresh mode)...")
    student = surgery.apply_surgery(student, teacher_weights, init_mode="fresh")

    # Create dual model
    logger.info("Creating dual attention model...")
    dual_model = DualAttentionModel(teacher, student)

    # Count params
    total_params = count_params(student.parameters())
    trainable_params = count_params(surgery.get_trainable_params(student, include_vo=True))

    logger.key_value({
        "Teacher params": f"{teacher_params:,}",
        "Student total params": f"{total_params:,}",
        "Student trainable params": f"{trainable_params:,}",
        "Semantic dim": sem_dim,
        "Geometric dim": geo_dim,
        "Value dim": v_dim,
    }, title="Model Config")

    # Config
    config = DualAttentionConfig(
        data_path=data_path,
        max_steps=max_steps,
        lr=lr,
        output_match_alpha=output_match_alpha,
        lm_alpha=lm_alpha,
        **kwargs,
    )

    # Train
    trainer = DualAttentionTrainer(dual_model, config, surgery)
    trainer.train()

    logger.header("Experiment Complete")
