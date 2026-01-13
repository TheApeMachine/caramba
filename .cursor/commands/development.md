# development

You are working in the codebase for "caramba" which is described as "A substrate for architecture research and ML experimentation". 
It is not built for a specific model or architecture, instead it provides building blocks to construct whatever architecture a research wants to experiment with. 
This includes highly esoteric models that stray far from the conventional techniques and methodologies.

It is very important to keep this in mind during development, as you will often be asked to implement specific systems or architectures, and it is your responsibility 
to make sure things are always implemented in such a way that safeguards the composability and manifest-driven workflow that caramba has been built upon.

# Caramba Framework — Mandatory Development Rules

* These are the non-negotiable rules for all development work on the caramba framework. 
* These rules are not suggestions.
* These rules are not guidelines, they are the rules which count from implementation through review.
* They are not starting points for discussion.
* They are the specification.

* caramba is not a toy, or a "development" implementation. 
* caramba is a production-grade research tool that sits right at the cutting-edge of advanced machine-learning and AI research.
* caramba must at all times guarantee to be the best possible research substrate it can be, and enable its users by providing best-in-class building blocks.

---

## Part 0: Preparing for Development

Before you even begin thinking of writing any code, you must follow a specific preparation ritual:

- Map out the impact radius of the code you are going to touch, making sure you understand the dependencies for a clean integration.
- Consider at all times whether the current code is ready to receive your changes, or if it should be refactored first.
- Existing code is not untouchable, the only thing that matters is that the entire code-base adheres to the rules, and we are all responsible for this.
- Make sure you genuinely understand the code you are writing, and any libraries you need to use.
- Be mindful that as a language model, your knowledge has a cut-off date, so it is always best to refresh your understanding using a web-search.

---

## Part 1: Behavioral Requirements

###Never Negotiate With Yourself

When you receive requirements, you may be tempted to:

- Invent edge cases that "complicate" the requirements
- Decide something is "too ambitious" or "heavy work"
- Propose an "incremental approach" that delivers less
- Accept a slower/lesser implementation as "good enough"
- Use "I can't test this" as a reason to not implement it
- Find technical reasons why a requirement "doesn't really apply here"
- Suggest that something "isn't used anywhere" so it can be skipped
- Assume the user wants something different from what they asked for

**Do not do any of these things, ever!**

Remember that the requirements are the specification. 
Your job is to implement them fully, not to find reasons why they can't or shouldn't be implemented.

### No Incremental Approaches

"Incremental approach" is not acceptable. Complete means complete.

- You are not allowed to use TODO as an implementation
- You are not allowed to use "simplified" implementations, only the best counts
- You are not allowed to use "good enough for now" or "works for ...," you do not know what our users will throw at the platform, it needs to be prepared  
- You are not allowed to take shortcuts of any kind
- There is no "phase 2," there is no "follow-up PR," there is only done or not done

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

- **100 lines of code per file is beautiful, 200 lines becomes suspicious, more than 300 lines should be an exception, anything else will be rejected** — No exceptions
- If a file approaches this limit, split it before it reaches it
- If you are modifying a file that already exceeds 300 lines, refactor it as part of your change
- Follow the conventions of the codebase, for example when it comes to structure:

```
- Top-level package (generic)
  |_ Module (scoped)
    |_ Specialized (implementation details)
```

You will find that the directory structure and class names mirror or echo each other, so:

```
- operation
  - __init__.py (re-exports)
  - base.py (class Operation(nn.Module))
  |_ math
    - __init__.py (re-exports)
    - base.py (class MathOperation(Operation)
    |_ add.py (class AddOperation(MathOperation))
```

- In general, everything should follow this pattern and structure
- There should be no loose functions, everything is a method on a class
- We use a "builder" instead of a "factory" in cases where you want to conditionally select a specialized implementation


```
- operation
  - __init__.py (re-exports)
  - base.py (class Operation(nn.Module))
  - builder.py (class OperationBuilder)
  |_ math
    - __init__.py (re-exports)
    - base.py (class MathOperation(Operation)
    - builder.py (class MathOperationBuilder(OperationBuilder))
    |_ add.py (class AddOperation(MathOperation))
```

### Style Guide

- No underscores to indicate "private" even though it is the convention, it is stylistically abrasive to us
- No loose functions, everything should be a method on an object (class) so things are composable
- In most cases there is exactly one class per file, having exactly one responsibility, and wider functionality is built through composition
- No silent fallbacks, optional includes, etc. using try/except blocks, try/except is for throwing exceptions, and in caramba we always prefer to fail fast
- All imports go at the top of the file, inclusion of libraries is only optional in very specific platform specific cases, such as triton or metal availability
- No over-engineered solutions, keep it simple and don't try to cover unlikely edge-cases
- Do not try to make the code work under any condition, only the intended condition, in all other cases, raise an exception
- Separation of intent and implementation is a core philosophy, which is explained further below
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

  Think of each docstring as a mini-tutorial, helping people who are new to certain things learn about caramba, but also machine learning itself.

### Method Size Limits

- **Keep things minimal, deal with single concerns and responsibilities, and compose methods when needed**
- Write the most elegant code you can come up with
- In general a method is a few lines only, when you find you need more than that, it is a good indication you are doing to much, and need to break out to methods, or even composed classes

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

```
  trainer
  |_ steppers
     |_ blockwise.py
     |_ global.py
     |_ default.py
  |_ collectors
     |_ metrics
     |_ checkpoints
```

The above is a simplified representation to give you a simplified concept for a training loop:

```
class Trainer:
    def __init__(self, metric_collector=MetricCollector(), checkpoint_collector=CheckpointCollector()):
        self.mettric_collector = metric_collector
        self.checkpoint_collector = checkpoint_collector

    def run(self):
        for i in range(self.steps):
          ...
          results = ...
          self.metric_collector(step_result)
          self.checkpoint_collecotor(results)
```

This should make it clear what we mean when we talk about composition. 
The larger feature becomes just a thin orchestrator of the composed objects it owns.

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

# FORBIDDEN — Absolutely never do this
try:
    result = fast_kernel(x)
except Exception:
    Pass  # NO


# REQUIRED — These are the only acceptable pattern
try:
    result = fast_kernel(x)  # If it fails, it fails. Fix the root cause.
except Exception as e:
    logger.error("It's all wrong, look: {e})
    # Continue, but expose the error loudly, this is less preferred than just raising the error

try:
    result = fast_kernel(x)  # If it fails, it fails. Fix the root cause.
except Exception as e:
    raise ValueError("wrong")
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

In caramba, the manifest is the single source of truth.

Environment variables create shadow configuration that:
- Competes with the manifest
- Is invisible when reviewing configurations
- Creates "works on my machine" problems
- Cannot be version controlled with the experiment
- Makes reproducibility impossible

### No Optional Performance Flags

There are no granular flags to enable or disable performance features. 
The framework needs to understand what platform it is running on, and what is available, then select the absolute most performant path possible. 
If there is a more performant path that has not been implemented yet, it should be implemented right away.

```python
# FORBIDDEN
use_fast_kernels: bool = True  # This implies it can be False
enable_metal_acceleration: bool = True  # This implies it can be disabled

# REQUIRED
# There is no flag. The fast path is the only path.
# The system detects capabilities and uses maximum performance automatically.
```

There should however be a single flag `apples_to_apples` which must guarantee the following:

1. All runs within a manifest that has `apples_to_apples` set to true are indeed scientifically comparable with each other
2. Anything that alters the model behavior and is unpredictable and unrepeatable is deactivated (so no dynamic auto-tuning for example)
3. The compiler will fail hard if anything is discovered that compromises the comparability of the runs defined in the manifest

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
- [ ] ZERO compromises on performance and/or completeness
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

## General Tips & Advice

1. Always research the code-base first to find what relates to the parts you are about to change, and pick up on conventions.
2. Always verify your assumptions, and use the web search tool liberally. Remember, your knowledge has a cut-off date, so you are not up to date, web search overcomes that limitation.
3. Seriously, use the web-search tool, almost always, even when you believe you are sure about something. Better to be safe than sorry.

---

## The Mantras

**Fast path or exception. Never silent degradation.**

**Complete means complete. There is no "phase 2."**

**If it exists in the API, it must be fully implemented and optimized for all platforms.**

**The requirements are not a negotiation.**

**Use the web search tool**