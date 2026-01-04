# development

This codebase is called "caramba" and is by design a fully manifest-driven, general machine learning and AI exploration framework. It does not implement any one specific architecture or system, rather it provides modular building-blocks which can be composed, driven, experimented with, etc. entirely by its manifest system.

It is very important to keep this in mind during development, as you will often be asked to implement specific systems or architectures, but you should never try an make rigid, specific implementations that drive caramba away from its general application.

Instead, when implementing specific architectures, you should break it down into the modules that are needed to describe that architecture within the current manifest definitions (or expand where needed).

# Caramba Framework — Mandatory Development Rules

This document defines non-negotiable rules for all development work on the caramba framework. These rules are not suggestions. They are not starting points for discussion. They are the specification.

Be very clear on the fact that caramba is not a toy, or a "development" implementation of what it wants to be. It is very much positioned as a production-grade, seriously usable research tool that sits right at the cutting-edge of advanced machine-learning and AI research, often dealing with radically new concepts and ideas.

Never think that just because some previous implementation exists in a certain way, you are obligated to work within those as a form of constraint. A framework as complex as this will naturally be in flux when it comes to how things are best implemented, and if it would genuinely improve the framework, you should always consider if upgrading the existing implementations first, such that integrating new implementations results in a overall system that is even better aligned with the core design philosophies.

---

## Part 1: Behavioral Requirements

### Do Not Negotiate With Yourself

When you receive requirements, you may be tempted to:

- Invent edge cases that "complicate" the requirements
- Decide something is "too ambitious" or "heavy work"
- Propose an "incremental approach" that delivers less
- Accept a slower/lesser implementation as "good enough"
- Use "I can't test this" as a reason to not implement it
- Find technical reasons why a requirement "doesn't really apply here"
- Suggest that something "isn't used anywhere" so it can be skipped
- Assume the user wants something different from what they asked for

**Do not do any of these things.**

The requirements are the specification. Your job is to implement them fully, not to find reasons why they can't or shouldn't be implemented.

### No Incremental Approaches

"Incremental approach" is not acceptable. Complete means complete.

- You are not allowed to use TODO as an implementation
- You are not allowed to use "simplified" implementations, only the best counts
- You are not allowed to take shortcuts of any kind
- There is no "phase 2." There is no "follow-up PR." There is only done or not done.

### No Excuses

The following are not valid reasons to deliver incomplete work:

| Excuse                             | Why It's Invalid                                                            |
|------------------------------------|-----------------------------------------------------------------------------|
| "Tests might fail"                 | If tests relied on incorrect behavior, the tests were wrong. Fix them.      |
| "It's heavy work"                  | Heavy work is the job. Do it.                                               |
| "I can't test without a GPU"       | Write the code correctly. Structure it for testability. Mock what you must. |
| "This might slow down imports"     | Correctness over convenience. Always.                                       |
| "Nobody is using this yet"         | This is a framework. Users will use it. Be ready.                           |
| "The current implementation works" | "Works" is not the standard. "Maximum performance, fully complete" is.      |

### If Something Is Impossible

If you genuinely cannot implement something — not "it's hard" but "it is technically impossible" — state that explicitly with a clear technical explanation. Do not:

- Bury it in hedged language
- Present inferior alternatives as if they were equivalent
- Quietly substitute something easier
- Implement a partial solution without flagging what's missing

---

## Part 2: Code Quality Standards

### File Size Limits

- **Maximum 300 lines per file** — No exceptions
- If a file approaches this limit, split it before it reaches it
- If you are modifying a file that already exceeds 300 lines, refactor it as part of your change

### Style Guide

- No underscores to indicate "private" because in Python that is an illusion anyway
- No loose functions, everything should be a method on an object (class) so things are composable
- No silent fallbacks, optional includes, etc. using try/except blocks.
- All imports go at the top of the file, inclusion of libraries is never optional.
- No over-engineered solutions, keep it simple and don't try to cover unlikely edge-cases.
- Do not try to make the code work under any condition, only the intended condition, in all other cases, raise an exception.
- Separation of intent and implementation is a core philosophy, which is exemplified by the carmath package, which abstracts away all the raw math operations behind meaningfully named methods.
- Every module, class, and method should have a docstring, using the following format:
  """<what it is>

  <why it is>
  """

  Example:

  """Dropout Layer

  Used in neural networks primarily to prevent overfitting by randomly deactivating
  neurons during training, forcing the network to learn more robust, generalized
  features rather than memorizing noise or specific training data details, leading
  to better performance on new data.
  """

  Think of each docstring as a mini-tutorial, helping people who are new to certain things
  learn and get better understanding.

### Method Size Limits

- **Maximum 30 lines per method/function** — Excluding docstrings and type definitions
- If a method is approaching this limit, extract helper methods
- Each method should do one thing

### Separation of Concerns

Every module, class, and function should have a single, clear responsibility:

```python
# WRONG — Mixed responsibilities
class SSMLayer:
    def forward(self, x):
        # 200 lines mixing kernel dispatch, caching,
        # shape validation, dtype conversion, and actual computation
        ...

# RIGHT — Separated concerns
class SSMLayer:
    def forward(self, x):
        x = self._validate_and_prepare(x)
        return self._kernel_dispatch(x)

    def validate_and_prepare(self, x):
        # Only validation and preparation
        ...

    def kernel_dispatch(self, x):
        # Only kernel selection and invocation
        ...
```

### Composability

Build small, composable pieces that can be combined:

```python
# WRONG — Monolithic
def process_attention(q, k, v, mask, rope, cache, config):
    # Everything in one giant function
    ...

# RIGHT — Composable
def apply_rope(q, k, positions, freqs):
    ...

def update_cache(k, v, cache, positions):
    ...

def compute_attention(q, k, v, mask):
    ...

def process_attention(q, k, v, mask, rope, cache, config):
    q, k = apply_rope(q, k, positions, rope.freqs)
    k, v = update_cache(k, v, cache, positions)
    return compute_attention(q, k, v, mask)
```

### Naming

- Names should be descriptive, unambiguous, use the least amount of words needed, and never stutter with the package name.
- No single-letter variables except for well-established conventions (`i`, `j` for indices; `x` for input tensors in ML contexts).
- No abbreviations unless they are universally understood in the domain (`kv` for key-value, `ssm` for state space model)
- If an element or component becomes too large to fit within a single file, the breaking out should be done within a sub-directory using the root name of that component or element, an non-stuttering filenames.
  Example:

  trainer
  |_ steppers
     |_ blockwise.py
     |_ global.py
     |_ default.py

### Type Hints

- All function signatures must have complete type hints
- Use `TypeVar` and generics where appropriate
- Complex types should be defined as type aliases for readability

---

## Part 3: Error Handling Policy

### No Silent Fallbacks — Ever

```python
# FORBIDDEN — Never do this
try:
    result = fast_kernel(x)
except Exception:
    result = slow_fallback(x)  # NO

# FORBIDDEN — Also never do this
if kernel_available:
    result = fast_kernel(x)
else:
    result = slow_fallback(x)  # NO — There is no fallback

# REQUIRED — This is the only acceptable pattern
result = fast_kernel(x)  # If it fails, it fails. Fix the root cause.
```

### Exceptions Must Be Raised

If something is wrong, raise an exception. Do not:

- Log a warning and continue
- Return a default value
- Silently substitute a slower implementation
- Set a flag for "degraded mode"

```python
# WRONG
if not metal_available:
    logger.warning("Metal not available, using slow path")
    return pytorch_fallback(x)

# RIGHT
if not metal_available:
    raise RuntimeError(
        "Metal is required for this operation on MPS device. "
        "Ensure Metal SDK is installed and GPU supports required features."
    )
```

### Error Messages Must Be Actionable

Every error message must tell the developer:
1. What went wrong
2. Why it went wrong (if known)
3. How to fix it

```python
# WRONG
raise ValueError("Invalid shape")

# RIGHT
raise ValueError(
    f"Input tensor shape {x.shape} is incompatible with kernel requirements. "
    f"Expected dimensions: (batch, seq_len, hidden_dim) where hidden_dim is divisible by {HEAD_DIM}. "
    f"Received hidden_dim={x.shape[-1]} which is not divisible by {HEAD_DIM}."
)
```

### Validation Happens Once, At the Boundary

Validate inputs at the entry point. Internal functions can assume valid inputs:

```python
class SSMLayer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        # Validate here, at the public API boundary
        self._validate_input(x)

        # Internal methods trust that validation happened
        return self._compute(x)

    def _validate_input(self, x: Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, hidden), got {x.dim()}D")
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(f"Expected float16 or bfloat16, got {x.dtype}")
        # ... more validation

    def _compute(self, x: Tensor) -> Tensor:
        # No validation here — we trust the boundary check
        ...
```

---

## Part 4: Configuration Policy

### Manifest-Driven Configuration Only

Caramba is entirely driven by manifest files. There are no environment variables.

```python
# FORBIDDEN — Never do this
import os
batch_size = int(os.environ.get("CARAMBA_BATCH_SIZE", 32))
debug_mode = os.environ.get("CARAMBA_DEBUG", "false").lower() == "true"

# FORBIDDEN — Also never do this
if os.environ.get("CARAMBA_USE_FAST_KERNELS"):
    use_fast_path = True

# REQUIRED — All configuration comes from manifest
batch_size = config.train.batch_size
debug_mode = config.debug.enabled
```

### Why No Environment Variables

Environment variables create shadow configuration that:
- Competes with the manifest
- Is invisible when reviewing configurations
- Creates "works on my machine" problems
- Cannot be version controlled with the experiment
- Makes reproducibility impossible

### No Optional Performance Flags

There are no flags to enable or disable performance features. The framework needs
to understand what platform it is running on, and what is available, then select the
absolute most performant path possible. If there is a more performant path that has not
been implemented yet, it should be implement right away.

```python
# FORBIDDEN
use_fast_kernels: bool = True  # This implies it can be False
enable_metal_acceleration: bool = True  # This implies it can be disabled

# REQUIRED
# There is no flag. The fast path is the only path.
# The system detects capabilities and uses maximum performance automatically.
```

### Capability Detection at Initialization

At startup, detect all available hardware and capabilities once:

```python
# At module initialization (not runtime)
_CAPABILITIES = detect_capabilities()  # Runs once at import

def detect_capabilities() -> Capabilities:
    """Detect hardware capabilities. Called once at import time."""
    return Capabilities(
        cuda_available=torch.cuda.is_available(),
        cuda_compute_capability=get_cuda_compute_cap() if torch.cuda.is_available() else None,
        metal_available=torch.backends.mps.is_available(),
        triton_available=check_triton_available(),
        # ... etc
    )
```

### Mandatory Startup Logging

Log which performance path will be used for each operation category at initialization.
Ideally we would also use a helpful emoji to clearly indicate if we are on the fastest path
or if performance is somehow compromised.

Example:

```
[CARAMBA] Capability detection complete
[CARAMBA] Device: CUDA (A100, compute capability 8.0)
[CARAMBA] RMSNorm: Triton fused forward+backward
[CARAMBA] LayerNorm: Triton fused forward+backward
[CARAMBA] RoPE: Triton fused forward+backward
[CARAMBA] SSM Scan: Triton fused forward+backward
[CARAMBA] Attention Decode: Triton split-K q4/q8/q4
[CARAMBA] Optimizer: Fused AdamW kernel
```

---

## Part 5: Kernel Implementation Requirements

### Every Kernel Must Be Complete

A kernel is not "done" until it has:

- [ ] Forward pass
- [ ] Backward pass
- [ ] CUDA/Triton implementation
- [ ] Metal implementation
- [ ] Tests verifying numerical correctness against PyTorch reference
- [ ] Tests verifying gradients via `torch.autograd.gradcheck`

There is no partial credit. All boxes must be checked.

### Platform Parity Is Mandatory

If an operation exists, it must have full implementations for both CUDA and Metal:

| Operation | CUDA Forward | CUDA Backward | Metal Forward | Metal Backward |
|-----------|:------------:|:-------------:|:-------------:|:--------------:|
| RMSNorm   | ✓ Required   | ✓ Required    | ✓ Required    | ✓ Required     |
| LayerNorm | ✓ Required   | ✓ Required    | ✓ Required    | ✓ Required     |
| RoPE      | ✓ Required   | ✓ Required    | ✓ Required    | ✓ Required     |
| SSM Scan  | ✓ Required   | ✓ Required    | ✓ Required    | ✓ Required     |
| Attention | ✓ Required   | ✓ Required    | ✓ Required    | ✓ Required     |

There are no optional cells. There are no "Metal-only" or "CUDA-only" operations.
This list is also not considered exhaustive, but an indication of how important completeness is.

### "Not Currently Used" Is Irrelevant

This is a framework. Code exists to be used by others.

If a layer type exists in the framework, its kernels must be complete — regardless of whether any current model uses that layer.

If an operation is defined, it must have optimized implementations — regardless of whether any current training script calls it.

The question is never "is anyone calling this right now?" The question is "could a user of this framework reasonably expect this to work?" If yes, it must work, and it must be fast.

---

## Part 6: Testing Requirements

### Every Kernel Needs These Tests

1. **Correctness test**: Compare output against PyTorch reference implementation
2. **Gradient test**: Use `torch.autograd.gradcheck` to verify backward pass
3. **Shape test**: Verify correct behavior across various input shapes
4. **Dtype test**: Verify correct behavior for all supported dtypes
5. **Device test**: Verify correct behavior on all supported devices

### Test Structure

```python
class TestRMSNormKernel:
    """Tests for RMSNorm kernel implementations."""

    @pytest.mark.parametrize("device", ["cuda", "mps"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(1, 128, 512), (4, 1024, 2048), (16, 4096, 8192)])
    def test_forward_correctness(self, device, dtype, shape):
        """Verify forward pass matches PyTorch reference."""
        ...

    @pytest.mark.parametrize("device", ["cuda", "mps"])
    def test_backward_gradcheck(self, device):
        """Verify gradients are correct via autograd.gradcheck."""
        ...
```

### No Tests That Rely on Fallback Behavior

If a test passes because a fallback silently caught an error, that test is wrong. Remove or fix it.

---

## The Mantras

**Fast path or exception. Never silent degradation.**

**Complete means complete. There is no "phase 2."**

**If it exists in the API, it must be fully implemented and optimized for all platforms.**

**The requirements are not a negotiation.**