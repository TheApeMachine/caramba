# Compiler

Compilation in Caramba means transforming a YAML manifest through multiple stages until it becomes an executable `Program` for the selected backend. The compiler is the bridge between human-authored configuration and machine-executable neural network operations.

## Overview

The compilation pipeline has three main phases:

```
YAML Manifest
    ↓
[Manifest Compilation] includes/substitutions/defaults
    ↓
Cap'n Proto Manifest Message
    ↓
[Topology Lowering] nested → flat graph
    ↓
GraphTopology
    ↓
[Runtime IR Building] graph → program
    ↓
Program (backend-agnostic IR)
    ↓
[Backend Compilation] IR → executable module
    ↓
TorchProgram / MLXProgram
```

## Phase 1: Manifest Compilation

**Location:** `framework/state/manifest/loader.py`

The first phase takes user-authored YAML and produces a canonical Cap'n Proto `Manifest` message. This involves:

1. **YAML Parsing** — Load the YAML file into a Python dictionary
2. **Include Expansion** — Resolve `!include` directives (planned)
3. **Variable Substitution** — Replace `${variable}` references with their values (planned)
4. **Default Application** — Fill in missing fields with sensible defaults
5. **Type Coercion** — Ensure all values match the expected Cap'n Proto types
6. **Message Construction** — Marshal the dictionary into a Cap'n Proto message

The loader is intentionally strict—it fails fast on any type mismatch or invalid structure. This ensures problems are caught immediately rather than during training.

```python
# Entry point
from caramba.framework.state.manifest.loader import load_path
manifest = load_path(path=Path("experiment.yml"))
```

### What Gets Compiled

The manifest contains:

- **Metadata** — name, description, URL, notes
- **Secrets** — API key references (values never stored in manifest)
- **Variables** — Reusable parameters (`d_model`, `vocab_size`, etc.)
- **Datasets** — Data source configurations
- **Trainer** — Global training configuration
- **Targets** — Execution targets (train, eval, predict)
  - Each target contains a `system` with `topology` and `embedder` specs

## Phase 2: Topology Lowering

**Location:** `framework/topology/` and `framework/state/manifest/compiler/topology/`

Topologies define the computational graph of the model. They describe how operations connect—which outputs feed into which inputs.

Higher-level topology types (like `ResidualTopology` or `NestedTopology`) must be "lowered" to the canonical `GraphTopology` format before they can be compiled to a program.

### GraphTopology Structure

The `GraphTopology` is the intermediate representation used by the runtime IR builder:

```yaml
topology:
  type: GraphTopology
  inputs: [x]  # Named input tensors
  nodes:
    - id: embed
      op: TokenEmbedding
      in: [x]
      out: [h]
      config:
        vocab_size: 50257
        d_model: 768
    - id: attn
      op: ScaledDotProductAttention
      in: [h, h, h]  # Q, K, V
      out: [attn_out]
      config:
        num_heads: 12
    # ... more nodes
  outputs: [logits]  # Named output tensors
```

Each node specifies:
- **id** — Unique identifier within the graph
- **op** — Operation type (maps to a class in `framework/operation/`)
- **in** — Input tensor names (references to inputs or other nodes' outputs)
- **out** — Output tensor names (can be referenced by downstream nodes)
- **config** — Operation-specific configuration

### Topology Types

| Type | Description | Status |
|------|-------------|--------|
| `GraphTopology` | Flat DAG of operations | Implemented |
| `ResidualTopology` | Adds skip connections | Planned |
| `NestedTopology` | Hierarchical blocks | Planned |
| `RepeatedTopology` | Repeated layers with weight sharing | Planned |

The topology builder (`framework/topology/builder.py`) selects the appropriate topology class based on the manifest.

## Phase 3: Runtime IR Building

**Location:** `framework/runtime/ir/builder.py`

The runtime IR builder takes a `GraphTopology` and produces a backend-agnostic `Program` Cap'n Proto message. This is the final representation before backend-specific compilation.

```python
from caramba.framework.runtime.ir.builder import build_program

program = build_program(manifest=manifest, target=target)
```

### Program Structure

A `Program` contains:

- **inputs** — Input tensor specifications (shapes, dtypes)
- **ops** — Ordered list of operations with their configurations
- **outputs** — Output tensor specifications

```python
# Internal structure (as dict before marshaling)
{
    "inputs": [],  # Shape inference fills these later
    "ops": [
        {
            "id": "embed",
            "type": "TokenEmbedding",
            "inputs": [{"name": "x"}],
            "outputs": [{"name": "h"}],
            "params": [
                {"name": "vocab_size", "value": {"intVal": 50257}},
                {"name": "d_model", "value": {"intVal": 768}},
            ],
        },
        # ... more ops
    ],
    "outputs": [{"name": "logits"}],
}
```

## Phase 4: Backend Compilation

**Location:** `framework/backend/program/`

The final phase converts the backend-agnostic `Program` IR into an executable module for the target backend (PyTorch, MLX, etc.).

Each backend has its own program class:

- `TorchProgram` — Wraps operations as a PyTorch module
- `MLXProgram` — Wraps operations for Apple Silicon execution

```python
from caramba.framework.backend.program.torch import TorchProgram

program = TorchProgram(operations=compiled_ops)
program.execute()
```

## Manifest Compiler Pipeline

**Location:** `framework/state/manifest/compiler/`

The manifest compiler is a chain of compilers that run in sequence. This allows modular, composable compilation stages:

```python
from caramba.framework.state.manifest.compiler.builder import ManifestCompilerBuilder

builder = ManifestCompilerBuilder(
    manifest_compilers=[
        IncludeExpander(manifest=manifest),
        VariableResolver(manifest=manifest),
        TopologyLowerer(manifest=manifest),
        Validator(manifest=manifest),
    ]
)
builder.build()  # Runs all compilers in order
```

### Adding a New Compiler Stage

1. Create a new class extending `BaseManifestCompiler`
2. Implement the `compile()` method
3. Add it to the compiler chain in the builder

```python
from caramba.framework.state.manifest.compiler.base import BaseManifestCompiler

class MyCompilerStage(BaseManifestCompiler):
    """My custom compilation stage"""
    
    def compile(self) -> None:
        # Mutate self.manifest as needed
        pass
```

## Validation

**Location:** `framework/state/manifest/validator/`

Validation happens as part of compilation. Validators check:

- **Structural validity** — Required fields present, correct types
- **Semantic validity** — References resolve, no cycles in graph
- **Sanity checks** — Warn about unusual values (e.g., learning rate > 1.0)

Validation fails hard and fast—no partial results, no silent degradation.

## RPC Integration

The compiler subsystem is designed to work over RPC using Cap'n Proto. This enables:

- Distributed compilation (compile on a different machine)
- Caching of compiled artifacts
- Separation of concerns between manifest authoring and execution

Key RPC services:
- `ManifestService` — Load, validate, compile manifests
- `TopologyService` — Lower topologies to `GraphTopology`
- `IrBuilderService` — Build `Program` from `GraphTopology`

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| YAML → Dict loading | ✅ Complete | `loader.py` |
| Dict → Cap'n Proto | ✅ Complete | `loader.py` |
| Include expansion | 🔲 Planned | Was in deleted `preprocessor.py` |
| Variable substitution | 🔲 Planned | Was in deleted `substitution.py` |
| Topology lowering | 🔲 Stub | `GraphTopologyCompiler` raises `NotImplementedError` |
| Runtime IR building | ✅ Complete | `framework/runtime/ir/builder.py` |
| Backend compilation | 🔲 Stub | Programs exist but minimal implementation |

## Future Work

1. **Include Expansion** — Support `!include path/to/file.yml` in manifests
2. **Variable Substitution** — Replace `${var}` with values from `variables` section
3. **Repeat Unrolling** — Expand `repeat: N` into N copies of a block
4. **Shape Inference** — Automatically compute tensor shapes through the graph
5. **Optimization Passes** — Fuse operations, eliminate dead code
6. **Caching** — Cache compiled programs by manifest hash
