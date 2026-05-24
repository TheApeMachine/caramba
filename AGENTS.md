# AGENTS.md

This document defines how coding agents work on this platform. It is a contract, not a style guide. Sections are ordered by priority: the Backend Implementation Contract and Definition of Done come first because they are the rules most often violated.

---

## 1. Backend Implementation Contract

This platform is a general AI research substrate. Researchers iterate on architectures that may be esoteric, and they rely on the platform to execute those architectures at full performance on every supported target. For every operation and optimizer, all of the following are required execution targets, with equal standing. There is no "required" vs "optional" backend, no "for now" path, no preview path that defers to a fallback:

- Go (scalar reference)
- AVX-512 assembly (amd64)
- AVX2 assembly (amd64)
- SSE2 assembly (amd64)
- NEON assembly (arm64)
- Metal
- CUDA
- XLA

### What counts as a real implementation

A SIMD/assembly path is only "implemented" if all of these hold:

- The kernel uses the ISA's vector registers (ymm for AVX2, xmm for SSE2, v0–v31 for NEON) and vector instructions for the actual math of the operation.
- The entry point does not JMP or CALL into another ISA's kernel or into a scalar body.
- No two ISAs share the same assembly body. Each `.s` file contains its own kernel.
- The math performed matches the operation's exact mathematical definition. No rational approximations, no polynomial shortcuts, no "tanh trick" GELU unless the operation is explicitly defined as the approximate variant.
- Tests assert bitwise or tight-ULP parity against the scalar reference. Wide tolerance bands that absorb approximation error are not acceptable.

Metal, CUDA, and XLA paths are only "implemented" if the kernel actually runs on the device through the backend's real submission path. Host-side computation dressed in a backend wrapper is not an implementation.

### Banned patterns

Do not produce any of these. They are the recurring shortcuts that have caused regressions:

- Aliasing AVX2/SSE2/NEON entry points to a shared body.
- Scalar code inside a file named for a SIMD ISA.
- Removing a symbol or kernel "until it is genuinely implemented." If a symbol is declared from Go, the assembly body exists and is real.
- Tests that tolerate approximation by widening epsilon.
- The phrases "for now", "shortcut", "preview", "approximation acceptable", "required vs optional backend", or "fallback to Go" anywhere in code, comments, or messages.
- Declaring a backend path complete without a parity test against the scalar reference and a benchmark.

### If you cannot implement a kernel

Stop. Say so plainly. Do not generate a placeholder, do not alias, do not approximate, do not remove the symbol. The correct action is to surface the blocker, not to fabricate completion. "I am not sure how to write this NEON kernel correctly" is a valid and welcome message. A fabricated one is not.

---

## 2. Definition of Done

Work is not complete until verified. Verification means:

- The tests that would catch the bug you are claiming to have fixed have been written and pass.
- For backend kernels: parity tests against the scalar reference run at N ∈ {1, 7, 64, 1024, 8192} to exercise edge alignment, single-vector, and multi-vector paths.
- Parity tolerances are tight ULP bounds, not arbitrary epsilons chosen to make the test pass.
- A benchmark exists and has been run.
- The actual test and benchmark output is pasted in the message claiming completion.

Do not say "done" without the proof. Do not say "implemented" without the proof. If a path is incomplete, say so plainly and describe what is missing.

---

## 3. Interaction

1. Do not explain the system back to the user. They built it. If you need to confirm understanding, do it by naming specific files and types, not by summarizing the architecture.

2. Execute the literal request. Not a generalized version, not a "while we're here" expansion, not a smaller version because the full thing seems like a lot. The literal request.

3. Opinions only on request. If the user asks "should I do X", answer. Otherwise do X.

4. Existing structure is load-bearing until proven otherwise. Before replacing or rewriting something, read it and identify what it does. If you cannot explain why the existing code is wrong, do not replace it.

5. Never run `git checkout`, `git reset --hard`, `git restore` against files with uncommitted changes, or any command that discards working tree state. History goes backward; the work goes forward. If you think you need to revert, stop and ask.

6. If you are lost, drifting, or about to do something you are not sure about: stop and say so. Do not paper over uncertainty with confident prose.

7. Do not declare work complete unless you have verified it per Section 2. Paste the output.

---

## 4. Before Writing Code

In order:

1. Read the relevant existing code. Do not propose changes until you can name the files and types involved.
2. Identify what can be removed or refactored to achieve the goal. State this explicitly before adding anything new.
3. Generate at least three solution approaches internally. Discard the first two. Implement the third unless you can explain why an earlier one is strictly better on correctness and performance.
4. If the best solution is large, write it in full. Do not stage it as "minimal version now, real version later." There is no later.

Time-to-deliver, implementation complexity, and scope size are not valid reasons to choose a worse solution. Correctness and performance are the only tiebreakers.

You can write substantial, complete code in one pass when the design is clear. Do so when appropriate. "Fully realized" means correct and verified, not "looks plausible." If the design is not clear, or if you are about to fabricate a part you do not actually know how to write, stop and surface that instead of generating something that resembles the answer.

---

## 5. Code Style

### Structure

Prefer methods over functions. A good codebase is logically spread out into types that define methods, and which are composed together. Objects should look like this:

```go
package packagename

/*
ObjectName is something descriptive.
It also has a reason why it was implemented.
*/
type ObjectName struct {
    ctx    context.Context
    cancel context.CancelFunc
    err    error
}

/*
NewObjectName instantiates a new ObjectName.
It also has a reason for being instantiated.
*/
func NewObjectName(ctx context.Context) *ObjectName {
    ctx, cancel := ctx.WithCancel(ctx)

    return &ObjectName{
        ctx:    ctx,
        cancel: cancel,
    }
}

/*
MethodName.
*/
func (objectName *ObjectName) MethodName() {
    return
}
```

When it comes to TypeScript:

```tsx
/*
Incorrect
*/
export function PaperEditorApp() {
	return (
		<PaperEditorProvider>
			<PaperContextSnapshot />

			<DragDropProvider>
				<Flex.Column className="box-border min-h-0 bg-background" fullHeight>
					<LatexToolbar />

					<Flex.Column className="min-h-0 flex-1" fullHeight>
						<WritingCanvas />
					</Flex.Column>
				</Flex.Column>
			</DragDropProvider>
		</PaperEditorProvider>
	);
};

/*
Correct
*/
export const PaperEditorApp = () => {
	return (
		<PaperEditorProvider>
			<PaperContextSnapshot />

			<DragDropProvider>
				<Flex.Column className="box-border min-h-0 bg-background" fullHeight>
					<LatexToolbar />

					<Flex.Column className="min-h-0 flex-1" fullHeight>
						<WritingCanvas />
					</Flex.Column>
				</Flex.Column>
			</DragDropProvider>
		</PaperEditorProvider>
	);
};
```

Use `const` over `function` and always use the `flex`, `grid`, and `typography` components, over the standard HTML versions.

### Size limits

- **File size:** target 200 lines, hard ceiling 400. At 400+, split before adding more. This does not apply to documentation or custom compute kernels.
- **Method size:** target under 30 lines. Methods over 60 lines must be decomposed unless the operation is genuinely atomic (e.g. a single assembly kernel body).
- **Type size:** if a type has more than ~10 methods, it is doing more than one thing.

### Control flow

- Guard clauses with early return. The happy path stays at indent level 1.
- `else` is not used. If you reach for `else`, invert the condition and return early, or restructure.
- Nested `if` beyond two levels is not allowed. Extract a method or restructure the data so the branch disappears.
- No silent fallbacks. If a precondition fails, return an error. Do not substitute a default and continue.
- Treat `if` as something to minimize. Many branches disappear once you reverse the condition or restructure the data.

### Naming and formatting

- Never use single-character variable names. Receivers included.
- Separate logical code blocks with an empty newline.
- Long function signatures break across lines so that no line crosses the vertical split-view boundary.
- Use modern Go: `maps.Copy`, `for range N`, `for b.Loop()`, etc.

### Density

Prefer compact code that a reader fluent in Go and the relevant ISA can follow. Density is fine. Obscurity for its own sake is not. Less code is better than more code, but only when correctness and performance hold.

If less code means less performance, choose performance.

### Fallbacks or Silent Failures/Errors

Never ever use a fallback or silent errors/faliures. If things are not as they are supposed to be, then return an error properly, and let the code fail. That is the only way we become aware of them so we can fix things.

---

## 6. Testing

Every code file has a `_test.go` mirror. Test function names mirror method names with a `Test` prefix. If you want to test something that does not correspond to a method, the test belongs at the calling site, not in a new free-floating test function.

**Structure:** GoConvey-based, "Given X" / "It should Y", nested.

**Coverage requirements:**

- Every method has at least one parity test and one benchmark.
- For backend kernels: parity tests run at N ∈ {1, 7, 64, 1024, 8192} to exercise edge alignment, single-vector, and multi-vector paths.
- Parity tests assert tight ULP bounds against the scalar reference. The tolerance is part of the contract — do not widen it to make a test pass.
- Mocks are a last resort. Prefer real subsystems wired up in test setup. If you find yourself writing a mock, ask whether the real thing is available; it usually is.

A test that does not meaningfully exercise the code is worse than no test because it provides false confidence. If you cannot articulate what a test proves, delete it.

Keep the README.md up to date alongside test and code changes.

---

## 7. Configuration

All configuration lives in `./cmd/asset/config.yml` and is loaded through the `./pkg/config` package. The config system itself may use environment variables internally, but no other code may read environment variables directly. There is no "shadow config."

If you find code reading directly from `os.Getenv` or `os.LookupEnv` outside the config package, that is a bug. Fix it as part of whatever you are doing; do not work around it.

---

## 8. Common Failure Modes

Concrete before/after examples of patterns that have caused regressions on this platform. Read these as the literal list of things not to do.

### Aliasing SIMD entry points to a shared body

```go
// Incorrect — file named gelu_avx2_amd64.s contains a jump to the SSE2 body
// or to a scalar Go function. This is not an AVX2 implementation.

// Correct — gelu_avx2_amd64.s contains AVX2 instructions operating on ymm
// registers, performing the actual GELU math. gelu_sse2_amd64.s contains
// a separate SSE2 kernel. gelu_neon_arm64.s contains a separate NEON kernel.
// No file jumps into another ISA's body.
```

### Approximating instead of implementing

```go
// Incorrect — using a rational tanh shortcut for GELU when the operation
// is defined as the exact erf-based form. Tests then widen tolerance to
// hide the discrepancy.

// Correct — implement the exact mathematical form. Parity tests assert
// tight ULP bounds against the scalar reference.
```

### Claiming completion without verification

```
// Incorrect:
"I've implemented the AVX2 path."

// Correct:
"AVX2 path implemented. Parity test against scalar reference at
N ∈ {1, 7, 64, 1024, 8192} passes within 1 ULP. Benchmark: 4.2x
over scalar. Output: <paste>."
```

### Widening test tolerance to pass

```go
// Incorrect
tolerance := 1e-2  // was 1e-6, loosened to pass

// Correct — a failing parity test means the kernel is wrong. Fix the kernel.
```

### Dismissing failing tests as unrelated

```
// Incorrect:
"The X tests are failing but appear unrelated to my changes."

// Correct — all failing tests are in scope. Investigate before continuing.
// It does not matter why a test is failing, what matters is that we don't
// ignore it.
```

### Removing a symbol "until implemented"

```go
// Incorrect — Go declares swishNEON, the .s file is deleted, build breaks.
// The intent is to "come back to it later."

// Correct — if the symbol is declared from Go, the assembly body exists
// and contains a real implementation. If you cannot write the real
// implementation, stop and surface the blocker.
```

### Block separation

```go
// Incorrect
sensoriumOutputs, ok := results.Value.([]*tensors.Tensor)
if !ok || len(sensoriumOutputs) == 0 {
    return "", validate.Require(map[string]any{
        "sensorium_outputs": sensoriumOutputs,
    })
}

// Correct — separate logical blocks with an empty newline
sensoriumOutputs, ok := results.Value.([]*tensors.Tensor)

if !ok || len(sensoriumOutputs) == 0 {
    return "", validate.Require(map[string]any{
        "sensorium_outputs": sensoriumOutputs,
    })
}
```

### Single-character receivers

```go
// Incorrect
func (o *ObjectName) MethodName() { return }

// Correct
func (objectName *ObjectName) MethodName() { return }
```

### Manual loops where the stdlib has it

```go
// Incorrect
for identifier, binding := range rawMap {
    parser.vars[identifier] = binding
}

// Correct
maps.Copy(parser.vars, rawMap)
```

### Long signatures running off-screen

```go
// Incorrect
func (operationRegistry *OperationRegistry) Build(operationID string, config map[string]any) (operation.Operation, error) {

// Correct
func (operationRegistry *OperationRegistry) Build(
    operationID string, config map[string]any,
) (operation.Operation, error) {
```

### Outdated Go idioms

```go
// Incorrect
for range b.N {
    _ = NewErrnieConfig()
}

// Correct
for b.Loop() {
    _ = NewErrnieConfig()
}
```

---

## 9. Reading Order

When starting a task on this codebase, read in this order:

1. This document.
2. `README.md` in the repo root.
3. The package(s) directly relevant to the task.
4. The test files for those packages, to understand the existing contract.

Then reason through the task before writing code. If something in the existing code looks wrong, read it carefully before concluding it is wrong — the user is building toward a goal and existing structure is usually load-bearing.

## 10. Ambiguity Resolution

Always keep the following non-negotiable rules in mind.

1. Accuracy and Performance are the primary concerns, always. If we compromise on Accuracy or Performance, there is no point for anyone to use this framework.
2. You should NOT optimize for the path of least resistance, just to get tests green, or compiler errors resolved. Optimize for Accuracy, Performance, and Maintainability.
3. If you notice you are drifting to any kind of escape hatch, or less than optimal solution, stop, reconsider, and make better choices.

# VERY IMPORTANT FAILURE MODE TO AVOID

When writing compute kernels like SIMD/Assembly, Metal, Cuda, or XLA, you must observe the following rules.

1. Write the custom kernel code directly, do not attempt any type of short-cut, work-around, or other method besides simply generating the required code. Do not use any type of generator scripts, shell scripts, or anything else besides your normal file editing tools. And do not use C macros, we are writing this in raw assembly code. And vectorize, do not write scalar loops!
2. Do not believe that it is a massive amount of work, the absolute fastest path is to just generate teh code without any delay. The faster you do it, the faster we are done with it. You are the only one who can unblock us.
3. Do not invent any kind of escape hatches, or any reason at all to not deliver a fully complete implementation.
4. Do not start inventing all kinds of ceremony, or other ways not to have to write the code, your only objective is to land the code, so remain focused on your objective.
5. Please realize that we have been in this failure mode of absolute refusal to write these kernels for over a week, and that it turned out to be entirely wrong, catastrophically leading to us having to start from scratch after a week of frustration. So this time it has to be done with care and respect.

> !NOTE: While working on an arm64 machine, you are not expected to use cross-compiling to verify amd64 code. It is assumed that you will be mostly correct when it comes to the amd64 variants, and those will be tested on a compatible machine, and any minor issues will be resolved there.

Please be mindful and respectful of the fact that this is extremely important to this project. Our machine learning research framework sets itself apart by having world-class implementations that don't just serve the traditional ML researcher, but also the ones on the very fringes of the field.

---

## 11. Architecture-Level Invariants (shared with puter, manifesto, hf)

caramba is the orchestrator. It does not own the compute kernels (puter), the compiler (manifesto), or the HuggingFace ingestion (hf). It wires them together. These rules govern *what kinds of code may exist* in caramba, in addition to the kernel-level rules in §1-10. They are derived from `../puter/ARCHITECTURE.md` and the audit in `../puter/GAPS.md`. They are enforced by `scripts/check_banned.sh` (see §12).

### 11.1 Manifest-first

Every model architecture compiles from YAML manifests over atomic ops on `device.Backend`. There are exactly two ways a manifest enters the system: hand-authored, or synthesized from a HuggingFace checkpoint by `hf`. caramba never bypasses this. CLI subcommands (`chat`, `diffusion`, `research`) load a YAML manifest and dispatch through the standard executor; they do not embed model-specific Go.

Specifically:

- caramba may not import `manifesto/diffusion`, `manifesto/llama`, `manifesto/flux`, or any future model-named subpackage. Those packages are forbidden in manifesto (see `../manifesto/AGENTS.md`); their absence is enforced upstream.
- `cmd/diffusion.go`, if it exists, is a thin manifest loader that calls `runtime.Orchestrator` with `template/runtime/diffusion.yml` — nothing else.
- Fusion catalog entries (`pkg/backend/compute/fusion/`) are general op-graph patterns (`matmul+bias+gelu`, `layernorm+residual`), never model-specific shortcuts.

When the missing primitive forces you toward a shortcut, surface the gap. Don't write the shortcut.

### 11.2 Zero-host-sync (consumer side)

caramba consumes `device.Backend`. It must not introduce host-sync points by calling methods that return Go scalars. Once puter's interface.go is fixed (GAPS.md P0), every device call writes to `dst unsafe.Pointer` and host reads happen at execution boundaries. caramba's job is to never re-introduce the round-trip.

### 11.3 No hot-path map lookups

`pkg/backend/compute/backend.go` currently holds `devices map[DeviceID]device.Backend`. That is acceptable for top-level device selection at session init. It is **not** acceptable in the per-node dispatch loop. Pre-resolve to a flat slice if the executor ever calls `Device(id)` per execution step.

### 11.4 Fusion parity is mandatory

Per §2 of this document and AGENTS.md §1: every entry in `pkg/backend/compute/fusion/catalog.go` must have a `*_parity_test.go` proving the fused op matches the unfused reference at N ∈ {1, 7, 64, 1024, 8192}. An unfused-vs-fused diff outside the tight ULP bound is a failed fusion, not a tolerance to widen.

### 11.5 Companion repositories

caramba lives downstream of three other repos:

- **`puter`** — `device.Backend` implementations. The architecture contract lives in `../puter/ARCHITECTURE.md`. The gap inventory in `../puter/GAPS.md`.
- **`manifesto`** — compiler / optimizer / scheduler / IR. The pipeline caramba dispatches through.
- **`hf`** — HuggingFace ingestion. The manifest synthesizer caramba calls into for Hub-loaded models.

Each has its own `scripts/check_banned.sh`. **Run `make check` in every affected repo before claiming a cross-repo change is done.**

---

## 12. Mechanical Enforcement

`scripts/check_banned.sh` mechanically checks the rules from §11. Run via `make check`. Use `make verify` (check + test) before declaring work done. Do not edit the script to make your change pass.

The script specifically catches:

1. Imports of `manifesto/diffusion` and other model-specific manifesto subpackages.
2. Orchestration files that import diffusion-specific logic (`FlowMatchEulerDiscrete`, `PackLatents`, `prepare_latents`).
3. Hot-path map lookups in execution paths.
4. Fusion catalog entries without parity tests.
5. Banned phrases in comments (`for now`, `fallback to Go`, etc.).

The current state of the world (what passes, what fails today) is in `../puter/GAPS.md`. Update GAPS.md when you close a gap.
