"""Block-wise distillation training for model upcycling.

When converting a model to a new architecture (like standard attention → DBA),
we can't train the entire model at once—the student would diverge too far from
the teacher. Instead, we train one block at a time, freezing all other blocks.
This ensures each block learns to match its teacher counterpart before moving on.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import weakref

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from caramba.console import logger
from caramba.model.trace import Trace, TraceStop
from caramba.trainer.distill import DistillLoss


@dataclass
class BlockwiseConfig:
    """Settings that control blockwise training performance and behavior.

    These exist as a separate config to keep the BlockwiseTrainer constructor
    clean while still allowing fine-grained control over optimizations.
    """

    # Teacher output caching: The teacher is frozen, so its outputs are
    # deterministic for a given input. Caching avoids redundant forward passes.
    cache_teacher_outputs: bool = True
    max_cache_size: int = 100

    # Mixed precision: Use float16/bfloat16 for faster training on GPU.
    # Only applies to CUDA and MPS devices.
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float16

    # Gradient accumulation: Simulate larger batch sizes by accumulating
    # gradients over multiple forward passes before stepping the optimizer.
    accumulation_steps: int = 1

    # Truncated forward: Stop the forward pass early for blocks near the
    # start of the model. Experimental—requires model architecture support.
    use_truncated_forward: bool = False

    # Gradient clipping: if > 0, clip grad norm before optimizer step.
    grad_clip_norm: float = 0.0


class TeacherOutputCache:
    """LRU cache for teacher model outputs.

    The teacher model never changes during distillation, so running it twice
    on the same input produces identical outputs. This cache stores those
    outputs to skip redundant computation.

    IMPORTANT:
    We key the cache by the *Python Tensor object identity* (using `id(x)` plus
    a weakref finalizer), not by data pointers / shapes. Tensor allocators
    (CUDA/MPS) aggressively reuse storage; pointer-based keys can collide across
    different batches and silently return incorrect teacher outputs.

    This cache will hit when the *same Tensor object* is reused (e.g. training
    multiple blocks on the same batch). It is intentionally conservative.
    """

    def __init__(self, max_size: int = 100) -> None:
        """Create a cache with the given maximum entry count.

        When the cache is full, the least-recently-used entry is evicted.
        """
        self._cache: dict[int, list[Tensor]] = {}
        self._refs: dict[int, weakref.ref[Tensor]] = {}
        self._access_order: list[int] = []
        self._max_size = max_size

    def _key(self, x: Tensor) -> int:
        """Return a stable identity key for the lifetime of `x`."""
        return int(id(x))

    def _attach_finalizer(self, x: Tensor, key: int) -> None:
        """Ensure this cache entry is removed when `x` is freed."""

        if key in self._refs:
            return

        def _on_collect(_: weakref.ref[Tensor]) -> None:
            self._cache.pop(key, None)
            self._refs.pop(key, None)
            try:
                self._access_order.remove(key)
            except ValueError:
                logger.error("Failed to remove key from access order, continuing")

        self._refs[key] = weakref.ref(x, _on_collect)

    def get(self, x: Tensor, *, upto: int | None = None) -> list[Tensor] | None:
        """Retrieve cached outputs for an input tensor.

        Returns None on cache miss. On hit, updates the LRU order.
        """
        key = self._key(x)
        if key in self._cache:
            # Move key to the end (most-recently-used).
            try:
                self._access_order.remove(key)
            except ValueError:
                # Could happen if the access list got out of sync (best-effort LRU).
                logger.error(f"Failed to remove key from access order, continuing: {key}")
            self._access_order.append(key)
            outs = self._cache[key]
            if upto is None:
                return outs
            u = int(upto)
            if u <= 0:
                return []
            if len(outs) >= u:
                return outs[:u]
            return None
        return None

    def put(self, x: Tensor, outputs: list[Tensor]) -> None:
        """Store outputs in the cache, evicting old entries if needed.

        Outputs are cloned and detached so they don't hold onto the
        computation graph or get modified by later operations.
        """
        key = self._key(x)
        self._attach_finalizer(x, key)

        if len(self._cache) >= self._max_size and key not in self._cache:
            # Evict least-recently-used key; tolerate keys already removed via GC callback.
            while self._access_order and len(self._cache) >= self._max_size:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]
                    self._refs.pop(oldest_key, None)
                    break

        existing = self._cache.get(key)
        if existing is None or len(outputs) > len(existing):
            # Store the most complete prefix we have for this input.
            self._cache[key] = [o.detach().clone() for o in outputs]
        if key not in self._access_order:
            self._access_order.append(key)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._cache.clear()
        self._refs.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self._cache)


class BlockwiseTrainer:
    """Trains a student model one block at a time to match a frozen teacher.

    Blockwise training is essential for architecture changes like DBA upcycling.
    If we trained all blocks simultaneously, early blocks would produce bad
    inputs for later blocks, causing the whole model to diverge. By training
    block-by-block with all other blocks frozen, each block sees stable inputs.
    """

    def __init__(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        optimizer: Optimizer,
        loss: DistillLoss,
        predicate: Callable[[str, nn.Module], bool],
        trace_predicate: Callable[[str, nn.Module], bool] | None = None,
        config: BlockwiseConfig | None = None,
    ) -> None:
        """Set up blockwise training between a teacher and student model.

        The predicate function identifies which modules count as "blocks"—
        typically attention layers. Both models must have the same number
        of blocks, since we train them in corresponding pairs.
        """
        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer
        self.loss = loss
        self._predicate = predicate
        self._trace_predicate = trace_predicate or predicate
        self._teacher_blocks = self._collect_blocks(teacher)
        self._student_blocks = self._collect_blocks(student)

        if not self._teacher_blocks:
            raise ValueError("Teacher has no blocks matching predicate.")
        if len(self._teacher_blocks) != len(self._student_blocks):
            raise ValueError(
                f"Teacher/student block counts must match, got "
                f"{len(self._teacher_blocks)} and {len(self._student_blocks)}"
            )

        teacher_trace_points = [
            module
            for name, module in teacher.named_modules()
            if self._trace_predicate(name, module)
        ]
        student_trace_points = [
            module
            for name, module in student.named_modules()
            if self._trace_predicate(name, module)
        ]
        if not teacher_trace_points:
            raise ValueError("Teacher has no trace points matching trace_predicate.")
        if len(teacher_trace_points) != len(student_trace_points):
            raise ValueError(
                "Teacher/student trace point counts must match, got "
                f"{len(teacher_trace_points)} and {len(student_trace_points)}"
            )
        if len(student_trace_points) != len(self._student_blocks):
            raise ValueError(
                "Trace point count must match block count for blockwise distillation, "
                f"got trace_points={len(student_trace_points)} blocks={len(self._student_blocks)}"
            )

        # Trace objects hook into the model to capture intermediate outputs
        self._teacher_trace = Trace(teacher, predicate=self._trace_predicate)
        self._student_trace = Trace(student, predicate=self._trace_predicate)

        self.config = config or BlockwiseConfig()

        # Set up teacher caching if enabled
        self._teacher_cache: TeacherOutputCache | None = None
        if self.config.cache_teacher_outputs:
            self._teacher_cache = TeacherOutputCache(
                max_size=self.config.max_cache_size
            )

        self._accumulation_count = 0
        self._device_type = self._detect_device_type()
        self._active_block_index: int | None = None

    def _detect_device_type(self) -> str:
        """Determine which device the model is on for autocast compatibility.

        Autocast requires knowing the device type (cuda, mps, or cpu).
        """
        for param in self.student.parameters():
            if param.device.type == "cuda":
                return "cuda"
            elif param.device.type == "mps":
                return "mps"
        return "cpu"

    def block_count(self) -> int:
        """Return the number of trainable blocks in the model."""
        return len(self._student_blocks)

    def step(
        self,
        x: Tensor,
        *,
        block_index: int,
        accumulate: bool = False,
    ) -> Tensor:
        """Run one distillation step for a single block.

        This is the core training operation: freeze all blocks except the
        target, run both models, compute loss between their outputs at that
        block, and update the student's weights for that block only.

        Args:
            x: Input token batch
            block_index: Which block (0-indexed) to train
            accumulate: If True, don't step optimizer (for gradient accumulation)

        Returns:
            The loss value (detached, for logging)
        """
        self._set_block_trainable(block_index)

        upto = int(block_index) + 1 if bool(self.config.use_truncated_forward) else None
        t_outputs = self._get_teacher_outputs(x, upto=upto)
        s_outputs = self._get_student_outputs(x, upto=upto)

        t_out = self._select_output(t_outputs, block_index, kind="teacher")
        s_out = self._select_output(s_outputs, block_index, kind="student")

        # Compute loss, optionally with mixed precision
        if self.config.use_amp and self._device_type in ("cuda", "mps"):
            with torch.autocast(
                device_type=self._device_type,
                dtype=self.config.amp_dtype,
            ):
                loss = self.loss([t_out], [s_out])
        else:
            loss = self.loss([t_out], [s_out])

        # Scale loss when accumulating gradients
        if self.config.accumulation_steps > 1:
            loss = loss / self.config.accumulation_steps

        loss.backward()

        # Step optimizer only after accumulating enough gradients
        self._accumulation_count += 1
        should_step = (
            not accumulate and
            self._accumulation_count >= self.config.accumulation_steps
        )

        if should_step:
            # Optional gradient clipping for stability (esp. at block boundaries).
            clip = float(getattr(self.config, "grad_clip_norm", 0.0))
            if clip > 0.0:
                try:
                    block = self._student_blocks[int(block_index)]
                    torch.nn.utils.clip_grad_norm_(block.parameters(), max_norm=clip)
                except Exception:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accumulation_count = 0

        return (loss * self.config.accumulation_steps).detach()

    def _get_teacher_outputs(self, x: Tensor, *, upto: int | None) -> list[Tensor]:
        """Get teacher block outputs, using cache when possible.

        Since the teacher is frozen, caching avoids redundant forward passes
        when training multiple blocks on the same batch.
        """
        if self._teacher_cache is not None:
            cached = self._teacher_cache.get(x, upto=upto)
            if cached is not None:
                return cached

        self._teacher_trace.clear()
        self._teacher_trace.max_outputs = upto
        with torch.inference_mode():
            with self._teacher_trace:
                try:
                    _ = self.teacher(x)
                except TraceStop as e:
                    logger.trace(f"BlockwiseTrainer: Teacher forward stopped correctly: {e}")

        outputs = list(self._teacher_trace.outputs)
        self._teacher_trace.max_outputs = None

        if self._teacher_cache is not None:
            self._teacher_cache.put(x, outputs)

        return outputs

    def _get_student_outputs(self, x: Tensor, *, upto: int | None) -> list[Tensor]:
        """Get student block outputs with gradient tracking.

        Unlike teacher outputs, these need gradients for backpropagation.
        """
        self._student_trace.clear()
        self._student_trace.max_outputs = upto

        if self.config.use_amp and self._device_type in ("cuda", "mps"):
            with torch.autocast(
                device_type=self._device_type,
                dtype=self.config.amp_dtype,
            ):
                with self._student_trace:
                    try:
                        _ = self.student(x)
                    except TraceStop as e:
                        logger.trace(f"BlockwiseTrainer: Student forward (AMP) stopped correctly: {e}")
        else:
            with self._student_trace:
                try:
                    _ = self.student(x)
                except TraceStop as e:
                    logger.trace(f"BlockwiseTrainer: Student forward stopped correctly: {e}")

        outputs = list(self._student_trace.outputs)
        self._student_trace.max_outputs = None
        return outputs

    def _collect_blocks(self, model: nn.Module) -> list[nn.Module]:
        """Find all modules in the model that match the predicate.

        These are the "blocks" we'll train one at a time.
        """
        return [
            module
            for name, module in model.named_modules()
            if self._predicate(name, module)
        ]

    def _set_block_trainable(self, block_index: int) -> None:
        """Freeze the entire model except for one block.

        This ensures gradients only flow through the block we're training,
        keeping all other blocks stable.
        """
        if block_index < 0 or block_index >= len(self._student_blocks):
            raise ValueError(
                f"Invalid block index {block_index}, expected "
                f"0..{len(self._student_blocks) - 1}"
            )

        if self._active_block_index == int(block_index):
            return

        for param in self.student.parameters():
            param.requires_grad = False

        block = self._student_blocks[block_index]
        for param in block.parameters():
            param.requires_grad = True

        self._active_block_index = int(block_index)

    def _select_output(
        self,
        outputs: list[Tensor],
        block_index: int,
        *,
        kind: str,
    ) -> Tensor:
        """Pick one block's output from the traced outputs list.

        The trace captures outputs from all blocks; we select just the one
        we're training.
        """
        if block_index < 0 or block_index >= len(outputs):
            raise ValueError(
                f"{kind} outputs missing block {block_index}, "
                f"got {len(outputs)} outputs."
            )
        return outputs[block_index]

    def clear_cache(self) -> None:
        """Empty the teacher output cache.

        Call this when switching to new data to avoid stale cache hits.
        """
        if self._teacher_cache is not None:
            self._teacher_cache.clear()

    def flush_gradients(self) -> None:
        """Force an optimizer step with any accumulated gradients.

        Use this at the end of training to ensure no gradients are lost.
        """
        if self._accumulation_count > 0:
            clip = float(getattr(self.config, "grad_clip_norm", 0.0))
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accumulation_count = 0
