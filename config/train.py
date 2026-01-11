"""Training configuration for model upcycling.

Training happens in two phases: blockwise distillation (train each layer
individually to match the teacher) and global fine-tuning (train the whole
model on language modeling). This module defines the hyperparameters and
optimization settings for both phases.
"""
from __future__ import annotations

import enum

from typing import Literal
from pydantic import BaseModel

from caramba.config import (
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

    # Resume/skip controls (manifest-driven).
    # If true, automatically resume from the latest checkpoint for the run/phase if present.
    auto_resume: bool = True
    # If true, skip running a phase when a corresponding *_final.pt checkpoint exists.
    skip_if_final: bool = True

    # Teacher model settings
    teacher_ckpt: str | None = None
    teacher_rope_base: PositiveFloat | None = None
    teacher_rope_dim: PositiveInt | None = None

    # Upcycle initialization knobs (student only).
    # - "svd": initialize DBA Q/K projections via (randomized) SVD of teacher Q/K.
    # - "random": keep student's randomly-initialized DBA Q/K projections (control for Experiment 1).
    dba_init: Literal["svd", "random"] = "svd"

    # Teacher sanity checks (upcycling safety).
    # These run once per target, immediately after loading the teacher checkpoint,
    # using a few batches from the configured token dataset.
    teacher_sanity_check: bool = True
    teacher_sanity_batches: PositiveInt = 2
    teacher_sanity_batch_size: PositiveInt = 1
    teacher_sanity_max_nll: PositiveFloat = 20.0
    # Optional stricter cap expressed in perplexity units. This is the guard that
    # catches "teacher is garbage but still finite" failures (e.g. wrong tokenizer
    # producing plausible-but-awful NLL like ~9 => ppl ~7k).
    teacher_sanity_max_ppl: PositiveFloat = 200.0
    # Optional reference check against HuggingFace transformers (gold baseline).
    # This avoids hardcoding an absolute ppl threshold that depends on dataset mix.
    # If enabled, we compute teacher NLL on the same batch in HF and require our
    # teacher to match within a small tolerance.
    teacher_sanity_reference: Literal["none", "hf"] = "none"
    teacher_sanity_ref_batches: PositiveInt = 1
    teacher_sanity_max_ppl_ratio_vs_ref: PositiveFloat = 1.25
    teacher_sanity_max_nll_delta_vs_ref: PositiveFloat = 0.25
    teacher_sanity_reference_fail_fast: bool = True

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
    # Gradient clipping (L2 norm). 0 disables.
    #
    # Relationship to `blockwise_grad_clip_norm`:
    # - During the BLOCKWISE phase, `blockwise_grad_clip_norm` applies per-block clipping.
    # - During the GLOBAL/STANDARD phase, `grad_clip_norm` applies global L2-norm clipping.
    #
    # Both `blockwise_grad_clip_norm` and `grad_clip_norm` may be set, but only the
    # clipping mode active for the current training phase will be used. Precedence is
    # phase-specific: if both are non-zero, the phase-specific parameter takes effect.
    # If training switches between phases, the respective parameter is used after the
    # switch.
    grad_clip_norm: NonNegativeFloat = 0.0
    num_workers: NonNegativeInt = 0
    pin_memory: bool = False
    prefetch_factor: PositiveInt = 2
    compile_model: bool | str = False
    compile_mode: str = "reduce-overhead"

    # Optimizer configuration (standard training loops).
    optimizer: str = "adamw"  # adamw|sgd|lion
    weight_decay: NonNegativeFloat = 0.0
    # "Intelligent" default: request fused optimizer when available, but allow
    # runtime to fall back (with loud warning) when unsupported on the current
    # device/dtype.
    fused_optimizer: bool = True
    offload_optimizer: bool = False

    # Distributed training (cluster promotion; optional).
    distributed_strategy: str = "none"  # none|ddp|fsdp
    distributed_backend: str = "nccl"  # nccl|gloo

    # Telemetry/profiling (optional; enabled explicitly via config).
    telemetry_interval: PositiveInt = 10
    profile_every: NonNegativeInt = 0
    profile_record_shapes: bool = False

    # Lightweight viz payloads (optional; disabled by default for performance).
    # When enabled, attention layers may copy small downsampled tensors back to CPU.
    viz_interval: NonNegativeInt = 0
    viz_tokens: PositiveInt = 16
    viz_channels: PositiveInt = 32
    viz_heads: PositiveInt = 4
    viz_topk: PositiveInt = 8

    # Per-layer activation stats telemetry (optional; disabled by default).
    # When enabled, forward hooks compute scalar stats and will synchronize devices
    # on the logging steps.
    layer_telemetry_interval: NonNegativeInt = 0

    # Activation checkpointing: controls when we recompute activations instead
    # of storing them to reduce peak memory usage on long sequences.
    activation_checkpointing: bool = False
    activation_checkpoint_threshold_mb: NonNegativeFloat = 0.0

    # LR scheduler (optional).
    scheduler: str = "none"  # none|linear|cosine|constant
    warmup_steps: NonNegativeInt = 0
    min_lr_ratio: Probability = 0.0

    # -----------------------------
    # MOSAIC curriculum controls (Stage D2 scheduled sampling)
    # -----------------------------
    # Teacher mixing probability for memory actions (when teacher signals are present in batch).
    # The trainer computes a per-step p_t and passes it through ctx.
    memblock_teacher_p_start: Probability = 1.0
    memblock_teacher_p_end: Probability = 0.0
    memblock_teacher_p_schedule: Literal["linear", "cosine", "constant"] = "linear"
    memblock_teacher_p_warmup_steps: NonNegativeInt = 0
    memblock_teacher_p_cooldown_steps: NonNegativeInt = 0

    # -----------------------------
    # MOSAIC write warmup (garbage-control experiments)
    # -----------------------------
    # Experimental / research flag (not production-ready): if > 0, disable memory
    # writes for the first N *training steps* by forcing write_mask=0 in the MOSAIC
    # block. This isolates the effect of early random writes ("garbage") from the
    # rest of learning. Default: 0 (no warmup; writes enabled immediately).
    #
    # Edge cases:
    # - If `memblock_write_warmup_steps` exceeds total training steps, writes will be
    #   disabled for the entire run.
    #
    # Interaction with `memblock_teacher_p_warmup_steps`:
    # - Both warmups are keyed off the same global training step and run concurrently.
    # - During the overlap, teacher-mixing p_t is still scheduled as configured, but
    #   writes remain forced-off until `memblock_write_warmup_steps` elapses.
    memblock_write_warmup_steps: NonNegativeInt = 0

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
