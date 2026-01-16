"""MLX trainers."""

from __future__ import annotations

from caramba.trainer.mlx.routing_hypothesis import RoutingHypothesisTrainer
from caramba.trainer.mlx.attention_distillation import AttentionDistillationTrainer
from caramba.trainer.mlx.dual_attention import DualAttentionTrainer

__all__ = ["RoutingHypothesisTrainer", "AttentionDistillationTrainer", "DualAttentionTrainer"]
