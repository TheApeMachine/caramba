# Program execution

How a program manifest runs end-to-end: package boundaries, import direction, and the step sequence.

---

## Package roles

| Package | Role |
|---------|------|
| **manifesto** | Language + orchestration: embedded assets, compile pipeline, IR orchestrator, program interpreter |
| **puter** | Device engine: pool discovery, graph runner, `device.Backend` kernels |
| **hf** | Hugging Face surface: Hub client, config ingestion, host IO, tokenizer |
| **caramba** | Dumb shell: CLI, frontend API, config loading — composes packages at the entrypoint |

caramba does not compile manifests, choose devices, dispatch kernels, or embed recipe templates. It instantiates the manifesto orchestrator and passes host-injected dependencies (Hub cache dir, stdin, puter runner).

---

## Import direction (no cycles)

```
caramba ──► manifesto
caramba ──► hf
caramba ──► puter

manifesto ──► hf          (Hub resolve, host IO/tokenizer)

puter ──► manifesto       (tensor, ir, runtime interfaces)

manifesto ──✗── puter     (forbidden — would cycle with puter → manifesto)
```

The orchestrator lives in **manifesto** even though placement and kernel dispatch are conceptually device work. Reason: manifesto already drives compile + program execution; if the orchestrator lived in puter, manifesto would have to call into puter while puter already imports manifesto.

**Split:**

| Concern | Package | Imports puter? |
|---------|---------|----------------|
| IR passes (verify, CSE, fusion, memory plan, placement *plan*) | `manifesto/runtime` | No |
| Interface definitions (`runtime.Backend`, device pool contract) | `manifesto/runtime` | No |
| Plan execution (`device.Backend.Gelu`, …) | `puter/runner` | N/A (is puter) |
| Device discovery | `puter/pool` | No |

manifesto/runtime emits an `ExecutionPlan`. puter/runner consumes it via interfaces defined in manifesto. No import cycle.

---

## What moves out of caramba

| Today (caramba) | Moves to |
|-----------------|----------|
| `pkg/asset` embedded templates (`template/**`) | `manifesto/asset` |
| `pkg/tokenizer` | `hf/tokenizer` |
| Host IO / tokenizer runtime wiring | `hf` (implements `manifesto/runtime.HostOps`) |
| `pkg/backend/compute` (pool, fusion catalog) | `puter/pool`, `puter/fusion` |
| Compile + run logic in cmd / pkg/workload | `manifesto/runtime` |

caramba keeps: `cmd/`, `pkg/config/`, `pkg/backend/api/`, `frontend/`.

---

## Entrypoint

```go
// cmd/root.go — caramba stays thin
func runRoot(cmd *cobra.Command, args []string) error {
    orchestrator, err := runtime.NewOrchestrator(runtime.OrchestratorOptions{
        Hub:     hf.NewHubFromConfig(config.NewHubConfig()),
        Compute: puter.NewRunner(puter.NewPool(cmd.Context())),
        Stdin:   os.Stdin,
    })
    if err != nil {
        return err
    }
    return orchestrator.Run(cmd.Context(), programPath)
}
```

One object. One call. Everything after that is manifesto's responsibility.

`Orchestrator.Run(ctx, programPath)` loads the manifest from **manifesto/asset**, then runs steps 1–7 below.

---

## Starting point

A program manifest is **loaded** when the orchestrator has resolved `programPath` to YAML bytes from `manifesto/asset` (or an absolute path override). Nothing executes until the orchestrator begins step 1.

---

## Step 1 — Flatten includes

**Where:** `manifesto/compiler` (`Compiler.CompileAssets` → `flattenIncludes`)

1. Read each `include:` from `manifesto/asset` or Hub cache (`hf/hub`).
2. Inline nested YAML into one document.
3. Resolve cross-manifest references.

**Output:** flattened YAML.

---

## Step 2 — Parse program IR

**Where:** `manifesto/parse` → `ast.Program`

**Output:** `ast.Program` — `main:`, `state:`, `schedulers:`, graph module refs. No devices.

---

## Step 3 — Compile referenced graphs

**Where:** `manifesto/compiler` (`CompileAssets`)

For each referenced model/graph: resolve → registry → expand → lower → `ir.Graph`.

**Output:** `Graphs`, `ComputeGraphs`.

---

## Step 4 — Discover devices

**Where:** `puter/pool` (injected into orchestrator via `Compute` / `Runner`)

Discover all available devices — `host:0`, `metal:*`, `cuda:*`, `xla:*`, `network:*` — each exposing `tensor.Backend` + `device.Backend`.

No hardcoded device. No CLI involvement.

---

## Step 5 — Build session + optimize

**Where:** `manifesto/runtime` + `manifesto/runtime`

Orchestrator assembles runtime state:

| Dependency | Source |
|------------|--------|
| Compiled program + graphs | manifesto compile (steps 1–3) |
| `HostOps` | `hf` (stdin, tokenizer encode/decode, image write) |
| `runtime.Backend` | `puter/runner` (injected) |
| State tensor residency | orchestrator placement → `puter/pool` |
| Schedulers | `manifesto/runtime` from program declarations |

Then, per `ir.Graph` (cached on first use):

1. Verify  
2. Canonicalize  
3. CSE / algebraic simplify  
4. Fusion (`puter/fusion` catalog consulted; plan emitted in manifesto)  
5. Memory plan  
6. Schedule / placement → `DeviceID` per node, explicit transfer edges  
7. Lower plan → typed `device.Backend` call descriptors (not string registry lookup)

**Output:** `ProgramSession` + cached `ExecutionPlan` per graph.

---

## Step 6 — Execute program steps

**Where:** `manifesto/runtime` (`Executor.Run`)

Walk `main:` sequentially:

| Op family | Handler |
|-----------|---------|
| `io.*`, `tokenizer.*` | `hf` via `HostOps` |
| `control.*`, `scheduler.*`, `state.*`, `value.*`, `sampling.*` | manifesto executor |
| `graph.call` | `puter/runner` via `runtime.Backend` |

Each `graph.call`: executor → runner → `ExecutionPlan` → `device.Backend.*` on assigned devices.

Chat and diffusion are YAML-only differences.

---

## End-to-end flow

```
caramba cmd/root.go
        │
        ▼
manifesto.Orchestrator.Run(programPath)
        │
        ├── [1] Flatten includes        manifesto/asset + hf/hub
        ├── [2] Parse → ast.Program   manifesto/parse
        ├── [3] Compile → ir.Graph    manifesto/compiler
        ├── [4] Discover devices      puter/pool (injected)
        ├── [5] Session + IR passes   manifesto/runtime
        └── [6] Execute main:         manifesto/runtime
                  │
                  ├── host ops ──► hf
                  └── graph.call ──► puter/runner ──► device.Backend.*
```

---

## Status

| Piece | Location | Status |
|-------|----------|--------|
| Compile pipeline | manifesto/compiler | Done |
| Embedded assets | manifesto/asset | Done |
| Tokenizer + host IO | hf/tokenizer, hf/program | Done |
| Orchestrator | manifesto/runtime | Done (compile + session; IR passes pending) |
| Device pool | puter/pool | Done |
| Graph runner | puter/runner | Stub (`CallGraph` not implemented) |
| caramba entrypoint | `Orchestrator.Run` | Done |

---

## Rules

1. **caramba** composes; it does not execute.
2. **manifesto/runtime** owns end-to-end flow after `Run(programPath)`.
3. **manifesto** must not import **puter** — use injected interfaces.
4. **puter/runner** executes plans via **`device.Backend` method calls**, not orchestrator-level string registries.
5. **No hardcoded device** anywhere in CLI, orchestrator defaults, or program YAML.
