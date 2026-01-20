"""MLX Trainer with Attention Distillation.

This trainer implements attention distillation for the routing hypothesis:
- Load pretrained Llama as teacher (frozen, provides target attention patterns)
- DBA model as student (fresh attention, learns to mimic teacher routing)
- Loss = LM loss + alpha * attention_distillation_loss

The key insight: DBA doesn't need to know which dimensions are semantic vs geometric.
It just needs to produce attention weights that match the teacher's.
The bottleneck forces DBA to discover the optimal decomposition.
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
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np

from adapter.mlx.surgery import AttentionSurgeryMLX
from console import logger
from layer.mlx.transformer import DBATransformer
from layer.mlx.standard_attention import TeacherModel


@dataclass
class DistillConfig:
    """Configuration for attention distillation training."""

    # Data
    data_path: str
    block_size: int = 2048
    batch_size: int = 1

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 5000
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0

    # Distillation
    distill_alpha: float = 1.0  # Weight for attention distillation loss
    lm_alpha: float = 0.1  # Weight for language modeling loss
    distill_layers: list[int] | None = None  # Which layers to distill (None = all)

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


def build_gradient_mask(model: DBATransformer, *, include_vo: bool = True) -> dict:
    """Build mask: 1.0 for trainable attention params, 0.0 for others.

    Args:
        model: The DBA transformer model
        include_vo: If True, include V/O projections. Set False for copy_vo mode
                    where V/O are copied from teacher and should stay frozen.
    """
    def _build_mask(params: Any, prefix: str = "") -> Any:
        if isinstance(params, dict):
            return {k: _build_mask(v, f"{prefix}.{k}" if prefix else k) for k, v in params.items()}
        elif isinstance(params, list):
            return [_build_mask(v, f"{prefix}.{i}") for i, v in enumerate(params)]
        elif isinstance(params, mx.array):
            # Check if this is an attention parameter
            is_attention = "attention" in prefix or "gate" in prefix
            if is_attention:
                # In copy_vo mode, exclude v_proj and out_proj
                if not include_vo and ("v_proj" in prefix or "out_proj" in prefix):
                    return mx.zeros_like(params)
                return mx.ones_like(params)
            else:
                return mx.zeros_like(params)
        return params
    return _build_mask(model.parameters())


def attention_distill_loss(
    student_attn: list[mx.array],
    teacher_attn: list[mx.array],
    layer_indices: list[int] | None = None,
) -> mx.array:
    """Compute attention distillation loss.

    Uses MSE between student and teacher attention distributions.
    KL divergence could also work but MSE is simpler and often sufficient.

    Args:
        student_attn: List of (B, H, T, S) attention weights from student
        teacher_attn: List of (B, H, T, S) attention weights from teacher
        layer_indices: Which layers to include (None = all)

    Returns:
        Scalar loss value
    """
    if layer_indices is None:
        layer_indices = list(range(len(student_attn)))

    total_loss = mx.array(0.0)
    n_layers = 0

    for i in layer_indices:
        if i < len(student_attn) and i < len(teacher_attn):
            s = student_attn[i]  # (B, H, T, S)
            t = teacher_attn[i]  # (B, H, T, S)

            # MSE loss on attention distributions
            diff = s - t
            layer_loss = mx.mean(diff * diff)
            total_loss = total_loss + layer_loss
            n_layers += 1

    if n_layers > 0:
        total_loss = total_loss / n_layers

    return total_loss


class AttentionDistillationTrainer:
    """Trainer with attention distillation from teacher to student.

    The teacher model (standard Llama attention) provides target attention
    patterns. The student (DBA) learns to reproduce those patterns while
    also minimizing LM loss. This gives DBA a "hint" about where to look,
    dramatically speeding up convergence.
    """

    def __init__(
        self,
        student: DBATransformer,
        teacher: TeacherModel,
        config: DistillConfig,
        surgery: AttentionSurgeryMLX,
    ):
        self.student = student
        self.teacher = teacher
        self.config = config
        self.surgery = surgery

        # Freeze teacher completely
        # (In MLX we just don't include it in optimizer)

        # Build gradient mask for student (only Q/K routing params, not V/O)
        # In copy_vo mode, V/O are copied from teacher and should stay frozen
        self._grad_mask = build_gradient_mask(student, include_vo=False)

        # Optimizer for student attention only
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

        # State for mx.compile
        self._state = [student.state, self.optimizer.state]

        # Training state
        self.step = 0
        self.loss_history: list[float] = []
        self.distill_loss_history: list[float] = []
        self.lm_loss_history: list[float] = []
        self.best_loss = float("inf")

    def train(self) -> None:
        """Main training loop with attention distillation."""
        logger.header("Attention Distillation Training", f"{self.config.max_steps} steps")

        logger.key_value({
            "Batch size": self.config.batch_size,
            "Block size": self.config.block_size,
            "Learning rate": f"{self.config.lr:.2e}",
            "Distill alpha": self.config.distill_alpha,
            "LM alpha": self.config.lm_alpha,
            "Warmup steps": self.config.warmup_steps,
        }, title="Training Config")

        data_iter = iter(self.data_loader)
        start_time = time.time()

        with logger.progress_bar() as progress:
            task = progress.add_task(
                "[info]Distilling attention patterns...[/info]",
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

                # Get batch
                input_ids, targets = next(data_iter)

                # Forward pass with attention weights
                loss, distill_loss, lm_loss, grads = self._forward_backward(
                    input_ids, targets
                )

                # Mask gradients (only attention params)
                grads = tree_map(lambda g, m: g * m, grads, self._grad_mask)

                # Gradient clipping
                if self.config.grad_clip_norm > 0:
                    flat_grads = tree_flatten(grads)
                    total_norm_sq = mx.array(0.0)
                    for _, g in flat_grads:
                        if isinstance(g, mx.array):
                            total_norm_sq = total_norm_sq + mx.sum(g * g)
                    total_norm = mx.sqrt(total_norm_sq)
                    clip_coef = mx.minimum(
                        mx.array(1.0),
                        mx.array(self.config.grad_clip_norm) / (total_norm + 1e-6)
                    )
                    grads = tree_map(lambda g: g * clip_coef, grads)

                # Update student
                self.optimizer.update(self.student, grads)

                # Evaluate
                mx.eval(self._state, loss, distill_loss, lm_loss)

                loss_val = float(loss)
                distill_val = float(distill_loss)
                lm_val = float(lm_loss)
                self.step += 1

                # Track metrics
                self.loss_history.append(loss_val)
                self.distill_loss_history.append(distill_val)
                self.lm_loss_history.append(lm_val)

                if loss_val < self.best_loss:
                    self.best_loss = loss_val

                # Progress update
                elapsed = time.time() - start_time
                tok_per_sec = (
                    self.step * self.config.batch_size * self.config.block_size / elapsed
                )
                ppl = math.exp(lm_val) if lm_val < 20 else float("inf")

                progress.update(
                    task,
                    completed=self.step,
                    description=(
                        f"[info]step {self.step}/{self.config.max_steps}[/info] "
                        f"[metric]loss={loss_val:.4f}[/metric] "
                        f"[muted]distill={distill_val:.4f} lm={lm_val:.4f}[/muted] "
                        f"[step]ppl={ppl:.1f}[/step] "
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
            "Final distill loss": f"{self.distill_loss_history[-1]:.4f}",
            "Final LM loss": f"{self.lm_loss_history[-1]:.4f}",
            "Final PPL": f"{math.exp(self.lm_loss_history[-1]):.2f}",
            "Best loss": f"{self.best_loss:.4f}",
            "Total time": f"{time.time() - start_time:.1f}s",
        }, title="Training Summary")

        self.save_checkpoint()

    def _forward_backward(
        self,
        input_ids: mx.array,
        targets: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, Any]:
        """Forward pass through both models, compute loss, backward through student.

        Returns:
            (total_loss, distill_loss, lm_loss, gradients)
        """
        # Forward through teacher (no grad needed)
        teacher_logits, _, teacher_attn = self.teacher(
            input_ids, return_attention=True
        )

        # Define loss function for student
        def student_loss_fn(model: DBATransformer):
            # Forward through student with attention
            student_logits, _, student_attn = model(
                input_ids, return_attention=True
            )

            # LM loss
            B, T, V = student_logits.shape
            lm_loss = nn.losses.cross_entropy(
                student_logits.reshape(-1, V),
                targets.reshape(-1),
                reduction="mean"
            )

            # Attention distillation loss
            if student_attn is None or teacher_attn is None:
                distill_loss = mx.array(0.0)
            else:
                distill_loss = attention_distill_loss(
                    student_attn,
                    teacher_attn,
                    layer_indices=self.config.distill_layers,
                )

            # Combined loss
            total_loss = (
                self.config.distill_alpha * distill_loss +
                self.config.lm_alpha * lm_loss
            )

            return total_loss, (distill_loss, lm_loss)

        # Compute gradients
        (total_loss, (distill_loss, lm_loss)), grads = nn.value_and_grad(
            self.student, student_loss_fn
        )(self.student)

        return total_loss, distill_loss, lm_loss, grads

    def save_checkpoint(self) -> None:
        """Save student attention checkpoint (only trainable Q/K params)."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Only save trainable params (Q/K routing, not V/O which are frozen)
        trainable_params = self.surgery.get_trainable_params(self.student, include_vo=False)
        path = save_dir / f"distill_checkpoint_{self.step}.npz"
        mx.savez(str(path), **trainable_params)
        logger.success(f"Checkpoint saved: {path}")


def load_teacher_from_llama(weights_path: str | Path) -> TeacherModel:
    """Load teacher model from Llama weights.

    Args:
        weights_path: Path to Llama safetensors/npz weights

    Returns:
        TeacherModel populated with Llama weights
    """
    weights_path = Path(weights_path)

    # Load weights
    loaded = mx.load(str(weights_path))
    if isinstance(loaded, dict):
        weights: dict[str, mx.array] = loaded
    elif isinstance(loaded, tuple) and len(loaded) >= 1:
        # Handle tuple return (dict, metadata)
        if isinstance(loaded[0], dict):
            weights = loaded[0]
        else:
            raise ValueError(f"Expected dict in tuple, got {type(loaded[0])}")
    else:
        raise ValueError(f"Unexpected return type from mx.load: {type(loaded)}")

    # Create teacher model with Llama 3.2 1B config (GQA: 32 Q heads, 8 KV heads)
    teacher = TeacherModel(
        d_model=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,  # Llama 3.2 1B uses GQA
        d_ff=8192,
        vocab_size=128256,
        rope_base=500000.0,
    )

    # Build weight mapping from Llama names to our model names
    weight_map: dict[str, mx.array] = {}

    # Token embeddings
    if "model.embed_tokens.weight" in weights:
        weight_map["embed_tokens.weight"] = weights["model.embed_tokens.weight"]

    # LM head
    if "lm_head.weight" in weights:
        weight_map["lm_head.weight"] = weights["lm_head.weight"]

    # Final norm
    if "model.norm.weight" in weights:
        weight_map["norm.weight"] = weights["model.norm.weight"]

    # Per-layer weights
    for i in range(16):
        llama_prefix = f"model.layers.{i}"
        our_prefix = f"layers.{i}"

        # Norms
        if f"{llama_prefix}.input_layernorm.weight" in weights:
            weight_map[f"{our_prefix}.norm1.weight"] = weights[f"{llama_prefix}.input_layernorm.weight"]
        if f"{llama_prefix}.post_attention_layernorm.weight" in weights:
            weight_map[f"{our_prefix}.norm2.weight"] = weights[f"{llama_prefix}.post_attention_layernorm.weight"]

        # Attention Q/K/V/O
        if f"{llama_prefix}.self_attn.q_proj.weight" in weights:
            weight_map[f"{our_prefix}.attention.q_proj.weight"] = weights[f"{llama_prefix}.self_attn.q_proj.weight"]
        if f"{llama_prefix}.self_attn.k_proj.weight" in weights:
            weight_map[f"{our_prefix}.attention.k_proj.weight"] = weights[f"{llama_prefix}.self_attn.k_proj.weight"]
        if f"{llama_prefix}.self_attn.v_proj.weight" in weights:
            weight_map[f"{our_prefix}.attention.v_proj.weight"] = weights[f"{llama_prefix}.self_attn.v_proj.weight"]
        if f"{llama_prefix}.self_attn.o_proj.weight" in weights:
            weight_map[f"{our_prefix}.attention.out_proj.weight"] = weights[f"{llama_prefix}.self_attn.o_proj.weight"]

        # FFN (SwiGLU)
        if f"{llama_prefix}.mlp.gate_proj.weight" in weights:
            weight_map[f"{our_prefix}.w_gate.weight"] = weights[f"{llama_prefix}.mlp.gate_proj.weight"]
        if f"{llama_prefix}.mlp.up_proj.weight" in weights:
            weight_map[f"{our_prefix}.w_up.weight"] = weights[f"{llama_prefix}.mlp.up_proj.weight"]
        if f"{llama_prefix}.mlp.down_proj.weight" in weights:
            weight_map[f"{our_prefix}.w_down.weight"] = weights[f"{llama_prefix}.mlp.down_proj.weight"]

    # Use tree_unflatten + update like the MLX examples do
    print(f"[DEBUG] Loading {len(weight_map)} weights into teacher model")
    print(f"[DEBUG] Sample keys: {list(weight_map.keys())[:5]}")
    teacher.update(tree_unflatten(list(weight_map.items())))
    mx.eval(teacher.parameters())  # Force evaluation like MLX example does

    return teacher


def run_attention_distillation(
    teacher_weights_path: str,
    data_path: str,
    *,
    sem_dim: int = 256,
    geo_dim: int = 512,
    v_dim: int = 768,
    max_steps: int = 5000,
    lr: float = 1e-4,
    distill_alpha: float = 1.0,
    lm_alpha: float = 0.1,
    **kwargs: Any,
) -> None:
    """Run attention distillation experiment.

    This is the main entry point for distilling Llama's attention patterns
    into DBA attention. The teacher provides target attention weights;
    the student learns to match them while also predicting next tokens.

    Args:
        teacher_weights_path: Path to pretrained Llama weights
        data_path: Path to tokenized training data (.npy)
        sem_dim: Semantic dimension for DBA
        geo_dim: Geometric dimension for DBA
        v_dim: Value dimension for DBA
        max_steps: Number of training steps
        lr: Learning rate
        distill_alpha: Weight for attention distillation loss
        lm_alpha: Weight for language modeling loss
        **kwargs: Additional config options
    """
    logger.header("Attention Distillation Experiment")

    # Load teacher model
    logger.info(f"Loading teacher from {teacher_weights_path}")
    teacher = load_teacher_from_llama(teacher_weights_path)
    teacher_params = count_params(teacher.parameters())
    logger.info(f"Teacher loaded: {teacher_params:,} params")

    # Create surgery adapter for student
    # For Llama retrofit we use a Llama-compatible DBA attention and compress semantic per-head.
    surgery = AttentionSurgeryMLX(
        sem_head_dim=8,
    )

    # Load teacher weights for surgery (to copy FFN/embeddings)
    teacher_weights = surgery.load_llama_weights(teacher_weights_path)

    # Create student model
    logger.info("Creating DBA student model...")
    student = surgery.create_dba_model()

    # Apply surgery (copy FFN/embeddings/V/O, fresh Q/K)
    # Using copy_vo mode: V and O projections are copied from teacher,
    # only Q/K (routing) are initialized fresh and trained.
    # This gives DBA the "known route" through FFN layers.
    logger.info("Applying attention surgery (copy_vo mode)...")
    student = surgery.apply_surgery(student, teacher_weights, init_mode="copy_vo")

    # Count params - in copy_vo mode, V/O are frozen
    total_params = count_params(student.parameters())
    trainable_params = count_params(surgery.get_trainable_params(student, include_vo=False))

    logger.key_value({
        "Teacher params": f"{teacher_params:,}",
        "Student total params": f"{total_params:,}",
        "Student trainable params": f"{trainable_params:,}",
        "Trainable %": f"{100*trainable_params/total_params:.1f}%",
        "Semantic dim": sem_dim,
        "Geometric dim": geo_dim,
        "Value dim": v_dim,
        "Distill alpha": distill_alpha,
        "LM alpha": lm_alpha,
    }, title="Model Config")

    # Create config
    config = DistillConfig(
        data_path=data_path,
        max_steps=max_steps,
        lr=lr,
        distill_alpha=distill_alpha,
        lm_alpha=lm_alpha,
        **kwargs,
    )

    # Train
    trainer = AttentionDistillationTrainer(student, teacher, config, surgery)
    trainer.train()

    logger.header("Experiment Complete")
