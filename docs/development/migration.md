Based on what I can see, here's my understanding of the current structure and where it needs to go:

## Current Structure

```
caramba/
├── framework/
│   ├── backend/
│   │   ├── base.py              # BaseBackend
│   │   ├── builder.py           # BackendBuilder (picks torch/mlx)
│   │   ├── torch.py             # TorchBackend
│   │   ├── mlx.py               # MLXBackend
│   │   ├── program/
│   │   │   ├── base.py          # BaseProgram
│   │   │   └── torch.py         # TorchProgram
│   │   └── rpc/
│   │       ├── client.py
│   │       ├── server.py        # BackendSelectorService
│   │       └── rpc.capnp
│   │
│   ├── operation/
│   │   ├── base.py              # Operation(nn.Module)
│   │   ├── activation/          # ReLU, LeakyReLU, Tanh, SwiGLU...
│   │   ├── attention/           # SDPA, etc.
│   │   ├── convolution/
│   │   ├── embedding/
│   │   ├── masking/
│   │   ├── math/
│   │   ├── normalization/
│   │   ├── positional/
│   │   ├── projection/
│   │   ├── pooling/
│   │   ├── shape/
│   │   └── tensor/
│   │
│   ├── runtime/
│   │   └── ir/
│   │       ├── builder.py       # build_program()
│   │       ├── loader.py
│   │       └── rpc/
│   │
│   ├── state/
│   │   ├── experiment/
│   │   │   ├── __init__.py      # ExperimentBuilder (stub)
│   │   │   └── builder.py
│   │   ├── manifest/
│   │   │   ├── loader.py
│   │   │   ├── compiler/
│   │   │   ├── template/        # YAML templates
│   │   │   └── *.capnp          # Schema definitions
│   │   ├── protocol/
│   │   │   └── ledger.capnp
│   │   └── notary.capnp         # Stub
│   │
│   ├── topology/
│   │   └── model.capnp
│   │
│   ├── cluster/
│   │   ├── discovery/
│   │   └── rpc/                 # ClusterService
│   │
│   ├── training/
│   │   └── optimizer/           # BaseOptimizer, AdamW, etc.
│   │
│   ├── telemetry/
│   │   └── ...                  # SSE streaming, activation hooks
│   │
│   └── helper/
│
└── research/
    ├── generator/               # Project scaffolding
    └── examples/
```

## What's Missing / Needs Restructuring

Based on our conversation, the architecture should be organized around **actors** (Cap'n Proto interfaces) that communicate via messages. Here's the target:

```
caramba/
├── actor/                           # ← NEW: Cap'n Proto actors (the core)
│   ├── notary/
│   │   ├── notary.capnp            # Interface + messages
│   │   ├── service.py              # Implementation
│   │   └── client.py               # Client wrapper
│   │
│   ├── experiment/
│   │   ├── experiment.capnp        # Interface + messages
│   │   ├── service.py
│   │   └── client.py
│   │
│   ├── model/
│   │   ├── model.capnp             # Interface + messages
│   │   ├── service.py
│   │   └── client.py
│   │
│   ├── storage/
│   │   ├── storage.capnp           # Interface + messages
│   │   ├── local.py                # LocalStorage impl
│   │   ├── s3.py                   # S3Storage impl
│   │   └── client.py
│   │
│   ├── backend/
│   │   ├── backend.capnp           # Interface + messages
│   │   ├── torch/
│   │   │   ├── service.py
│   │   │   └── compiler.py         # GraphTopology → nn.Module
│   │   ├── mlx/
│   │   │   ├── service.py
│   │   │   └── compiler.py
│   │   └── client.py
│   │
│   └── worker/                      # ← NEW: Remote execution
│       ├── worker.capnp            # Interface for remote training
│       ├── service.py
│       └── client.py
│
├── schema/                          # ← NEW: Shared data schemas
│   ├── manifest.capnp
│   ├── protocol.capnp
│   ├── architecture.capnp
│   ├── topology.capnp
│   ├── program.capnp
│   ├── checkpoint.capnp
│   ├── metrics.capnp
│   └── ledger.capnp
│
├── operation/                       # Operations (nn.Module implementations)
│   ├── registry.py                  # ← NEW: Explicit @register_op
│   ├── base.py
│   ├── activation/
│   ├── attention/
│   └── ...
│
├── compiler/                        # ← RESTRUCTURED
│   ├── manifest/                    # YAML → Manifest message
│   │   ├── loader.py
│   │   ├── include.py               # !include expansion
│   │   └── variable.py              # ${var} substitution
│   ├── architecture/                # Architecture → Topology
│   │   └── lowerer.py
│   └── topology/                    # Topology → Program
│       └── builder.py
│
├── training/                        # Training utilities (backend-agnostic)
│   ├── optimizer/
│   │   ├── base.py
│   │   └── adamw.py
│   ├── scheduler/
│   └── dataloader/
│
├── cluster/                         # Distributed coordination
│   ├── discovery/
│   ├── router.py                    # ← NEW: Routes jobs to backends
│   └── lease.py                     # ← NEW: Resource leasing
│
├── telemetry/
│
└── cli/                             # ← NEW: Command line interface
    ├── run.py                       # caramba run manifest.yml
    ├── status.py                    # caramba status
    └── inspect.py                   # caramba inspect model.cbm
```

## Key Changes

### 1. Actors as First-Class Citizens

Every major component becomes a Cap'n Proto interface:

```capnp
# actor/notary/notary.capnp
interface Notary {
  validateManifest @0 (manifest :Manifest, model :Model) -> (result :ValidationResult);
  validateCheckpoint @1 (checkpoint :Checkpoint, expected :Expected) -> (result :ValidationResult);
  validateFinal @2 (experiment :ExperimentRef) -> (result :ValidationResult);
  audit @3 (claim :Text, asOf :UInt64) -> (result :AuditResult);
}

# actor/experiment/experiment.capnp  
interface Experiment {
  getState @0 () -> (state :ExperimentState);
  executeProtocol @1 (protocol :Protocol) -> (run :RunRef);
  checkpoint @2 () -> (checkpoint :Checkpoint);
  commit @3 (approval :Approval) -> (model :ModelRef);
  void @4 (reason :Text) -> ();
}

# actor/backend/backend.capnp
interface Backend {
  capabilities @0 () -> (caps :BackendCapabilities);
  compile @1 (architecture :Architecture) -> (program :Program);
  spawn @2 (program :Program, config :RunConfig) -> (run :RunRef);
}
```

### 2. Separation of Schema and Implementation

```
schema/           # Pure data definitions (Cap'n Proto)
actor/            # Interfaces + implementations
operation/        # PyTorch/MLX nn.Module implementations
compiler/         # Transformation logic
```

### 3. Operation Registry with Explicit Registration

```python
# operation/registry.py
OPERATION_REGISTRY: dict[str, type[Operation]] = {}

def register_op(op_id: str):
    def decorator(cls):
        OPERATION_REGISTRY[op_id] = cls
        return cls
    return decorator

# operation/projection/linear.py
@register_op("projection.linear")
class LinearOperation(Operation):
    ...
```

### 4. Worker Actor for Distribution

```capnp
# actor/worker/worker.capnp
interface Worker {
  # Worker reports its capabilities
  getCapabilities @0 () -> (caps :WorkerCapabilities);
  
  # Accept a job
  accept @1 (job :Job) -> (run :RunRef);
  
  # Health check
  ping @2 () -> (timestamp :UInt64);
}

interface RunRef {
  # Stream events
  subscribe @0 (subscriber :RunSubscriber) -> ();
  
  # Control
  getMetrics @1 () -> (metrics :Metrics);
  pause @2 () -> ();
  resume @3 () -> ();
  cancel @4 () -> ();
}
```

## Migration Path

1. **Phase 1**: Create `schema/` with all Cap'n Proto data definitions
2. **Phase 2**: Create `actor/` structure, move RPC code there
3. **Phase 3**: Add operation registry, migrate operations to use `@register_op`
4. **Phase 4**: Restructure `compiler/` to be a clean pipeline
5. **Phase 5**: Implement `Experiment` and `Run` properly
6. **Phase 6**: Add `Worker` actor for distribution
