"""Training configuration for model upcycling.

Training happens in two phases: blockwise distillation (train each layer
individually to match the teacher) and global fine-tuning (train the whole
model on language modeling). This module defines the hyperparameters and
optimization settings for both phases.
"""
from __future__ import annotations

import enum

from pydantic import BaseModel

from config import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    Probability,
)


class TrainPhase(str, enum.Enum):
    """Which phase of training to run.

    BLOCKWISE trains each attention layer individually to match the teacher.
    GLOBAL fine-tune/train the whole model.
    STANDARD standard end-to-end training (non-upcycling).
    """

    BLOCKWISE = "blockwise"
    GLOBAL = "global"
    STANDARD = "standard"


class TrainConfig(BaseModel):
    """Hyperparameters and settings for a training run.

    The basics (batch_size, lr, device) are straightforward. The convergence_*
    fields enable adaptive training where each block trains until its loss
    drops below a target. The performance fields enable optimizations like
    teacher caching and mixed precision.
    """

    # Core training parameters
    phase: TrainPhase
    batch_size: PositiveInt
    block_size: PositiveInt
    lr: PositiveFloat
    device: str = "cpu"
    dtype: str = "float32"

    # Auto batch sizing: optionally scale batch size inversely with block_size.
    auto_batch_size: bool = False
    auto_batch_ref_block_size: PositiveInt = 512
    auto_batch_min: PositiveInt = 1

    # Teacher model settings
    teacher_ckpt: str | None = None
    teacher_rope_base: PositiveFloat | None = None
    teacher_rope_dim: PositiveInt | None = None

    # Convergence-based blockwise training: instead of fixed steps per block,
    # train each block until loss drops below target or patience runs out.
    convergence_target: PositiveFloat | None = None
    convergence_patience: PositiveInt = 50
    convergence_max_steps: PositiveInt = 5000

    # Blockwise "autopilot" (lightweight self-tuning)
    # This is intentionally simpler than the global orchestrator: it operates
    # within each block to adapt LR on spikes/plateaus and to log decisions.
    blockwise_autotune_enabled: bool = False
    blockwise_autotune_mode: str = "monitor"  # off|monitor|active
    blockwise_autotune_min_lr: PositiveFloat = 1e-6
    blockwise_autotune_lr_decay: PositiveFloat = 0.5  # multiply LR by this
    blockwise_autotune_plateau_patience: PositiveInt = 100
    blockwise_autotune_ema_decay: PositiveFloat = 0.99
    blockwise_autotune_spike_std: PositiveFloat = 3.0
    blockwise_autotune_window_size: PositiveInt = 50
    blockwise_autotune_log_every: PositiveInt = 50

    # Blockwise distillation details
    # - distill_target: "attention" matches raw AttentionLayer outputs (default behavior),
    #   "residual" matches the *post-residual* output of the attention ResidualTopology
    #   (often more stable for deeper layers).
    blockwise_distill_target: str = "attention"  # attention|residual
    blockwise_truncated_forward: bool = True
    blockwise_teacher_cache_max_size: PositiveInt = 100
    blockwise_grad_clip_norm: NonNegativeFloat = 0.0
    blockwise_reset_lr_each_block: bool = False

    # Performance optimizations
    cache_teacher_outputs: bool = True
    use_amp: bool = False
    amp_dtype: str = "float16"
    gradient_accumulation_steps: PositiveInt = 1
    num_workers: NonNegativeInt = 0
    pin_memory: bool = False
    compile_model: bool | str = False
    compile_mode: str = "reduce-overhead"

    # Optimizer configuration (standard training loops).
    optimizer: str = "adamw"  # adamw|sgd|lion
    weight_decay: NonNegativeFloat = 0.0
    fused_optimizer: bool = False
    offload_optimizer: bool = False

    # Distributed training (cluster promotion; optional).
    distributed_strategy: str = "none"  # none|ddp|fsdp
    distributed_backend: str = "nccl"  # nccl|gloo

    # Telemetry/profiling (best-effort; should not crash training).
    telemetry_interval: PositiveInt = 10
    profile_every: NonNegativeInt = 0
    profile_record_shapes: bool = False

    # Activation checkpointing: controls when we recompute activations instead
    # of storing them to reduce peak memory usage on long sequences.
    activation_checkpointing: bool = False
    activation_checkpoint_threshold_mb: NonNegativeFloat = 0.0

    # LR scheduler (optional).
    scheduler: str = "none"  # none|linear|cosine|constant
    warmup_steps: NonNegativeInt = 0
    min_lr_ratio: Probability = 0.0

    # Orchestrator settings (dynamic optimizer switching)
    orchestrator_enabled: bool = False
    orchestrator_mode: str = "active"  # disabled|monitor|active
    orchestrator_decision_interval: PositiveInt = 500
    orchestrator_eval_horizon: PositiveInt = 100
    orchestrator_initial_strategy: str = "conservative_adamw"
    orchestrator_use_adagc: bool = True
    orchestrator_adagc_warmup: NonNegativeInt = 100
    orchestrator_use_nowcasting: bool = False

    # Safety (used by GlobalOrchestratedStepper)
    orchestrator_max_loss_increase: PositiveFloat = 1.5
    orchestrator_max_spikes_before_switch: PositiveInt = 3
    orchestrator_safety_strategy: str = "spike_resistant"
    orchestrator_fail_fast: bool = True
    orchestrator_portfolio_base_lr: PositiveFloat = 1e-4
