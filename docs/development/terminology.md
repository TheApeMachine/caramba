```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                              MODEL (committed)                              │
│                        The auditable source of truth                        │
│                                                                             │
│   Contains EVERYTHING:                                                      │
│   - Architectures used                                                      │
│   - Weights (all variants)                                                  │
│   - Full training history                                                   │
│   - All metrics and benchmarks                                              │
│   - Comparative analysis                                                    │
│   - Generated artifacts                                                     │
│   - Protocols that governed execution                                       │
│   - Manifests that declared intent                                          │
│   - Complete ledger of events                                               │
│                                                                             │
│   Immutable once committed. Self-describing. Fully auditable.               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │
                              COMMIT (if verified)
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                          EXPERIMENT (transient)                             │
│                      Potential next state of the Model                      │
│                                                                             │
│   - Starts as a copy of current Model state                                 │
│   - Accumulates updates during execution                                    │
│   - Weights mutate in place                                                 │
│   - Metrics accumulate                                                      │
│   - Ledger grows                                                            │
│                                                                             │
│   NOT yet part of the Model. Can be voided without affecting Model.         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │
                              VOID (if validation fails)
                                      │
                                      ▼
                                   (discarded)
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ARCHITECTURE                                                                │
│ "What kind of model"                                                        │
│                                                                             │
│ Examples: StandardAttention, DBA, GQA, Bottleneck                           │
│ Defined in: architecture/*.yml                                              │
│ Contains: structural decisions, layer types, dimension ratios               │
├─────────────────────────────────────────────────────────────────────────────┤
│ TOPOLOGY                                                                    │
│ "How operations connect"                                                    │
│                                                                             │
│ Examples: GraphTopology, StackedTopology, ResidualTopology                  │
│ Defined by: lowering an Architecture to an executable graph                 │
│ Contains: nodes, edges, input/output bindings                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ OPERATION                                                                   │
│ "What computation happens"                                                  │
│                                                                             │
│ Examples: LinearOperation, RMSNormOperation, SDPAOperation                  │
│ Defined in: operation/**/*.py                                               │
│ Contains: forward(), parameters, config                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

```
Architecture (declarative, YAML)
      │
      │  "What structure do I want?"
      ▼
Topology (graph representation)
      │
      │  "How do operations connect?"
      ▼
Operations (executable, nn.Module)
      │
      │  "What computation happens?"
      ▼
Program (backend-agnostic IR)
      │
      │  "Serialize for provenance + backend dispatch"
      ▼
TorchProgram / MLXProgram (executable)
```

```py
class Model:
    """
    The committed, immutable, auditable artifact.
    Contains everything needed to reproduce and verify the research.
    """
    
    # What was built
    architectures: dict[str, Architecture]
    weights: dict[str, StateDict]        # architecture_id → weights
    
    # How it was built
    manifests: list[Manifest]            # Intent declarations
    protocols: list[Protocol]            # Execution rules
    
    # What happened
    ledger: Ledger                       # Append-only event log
    metrics: dict[str, Metrics]          # All recorded metrics
    benchmarks: dict[str, BenchmarkResult]
    analysis: Analysis
    
    # What was produced
    artifacts: list[Artifact]            # Tables, figures, etc.
    
    def audit(self, claim: str) -> AuditResult:
        """Verify any claim about this model's provenance."""
        ...


class Experiment:
    """
    Transient execution context. Potential next state of a Model.
    """
    
    def __init__(self, *, model: Model, manifest: Manifest, notary: Notary):
        # Start from current model state
        self.base_model = model
        self.manifest = manifest
        self.notary = notary
        
        # Working state (will become part of Model if committed)
        self.architectures: dict[str, Architecture] = {}
        self.weights: dict[str, StateDict] = {}
        self.ledger = Ledger()
        self.metrics: dict[str, Metrics] = {}
        self.protocols: list[Protocol] = []
        self.benchmarks: dict[str, BenchmarkResult] = {}
        self.analysis: Analysis | None = None
        self.artifacts: list[Artifact] = []
        
        # Execution state (not persisted to Model)
        self.runs: dict[str, Run] = {}
        self.optimizers: dict[str, Optimizer] = {}
    
    def execute(self) -> ExperimentResult:
        """Execute the manifest. Returns VERIFIED or VOIDED."""
        try:
            self._train_all()
            self._benchmark_all()
            self._analyze()
            self._generate_artifacts()
            
            # Final validation
            if self.notary.validate_final(self):
                return ExperimentResult.VERIFIED
            else:
                return ExperimentResult.VOIDED
                
        except Exception as e:
            self.ledger.append(FailureEvent(reason=str(e)))
            return ExperimentResult.VOIDED
    
    def commit(self) -> Model:
        """
        Merge experiment state into a new Model.
        Only called after VERIFIED result.
        """
        return Model(
            # Preserve history from base model
            manifests=self.base_model.manifests + [self.manifest],
            protocols=self.base_model.protocols + self.protocols,
            ledger=self.base_model.ledger.merge(self.ledger),
            
            # Add new state from this experiment
            architectures={**self.base_model.architectures, **self.architectures},
            weights={**self.base_model.weights, **self.weights},
            metrics={**self.base_model.metrics, **self.metrics},
            benchmarks={**self.base_model.benchmarks, **self.benchmarks},
            analysis=self.analysis,
            artifacts=self.base_model.artifacts + self.artifacts,
        )
```

The flow:

```
Model(v0) ──┐
            │
            ▼
     ┌─────────────┐
     │ Experiment  │◀── Manifest (intent)
     │             │◀── Protocols (rules)
     │  (working)  │
     └──────┬──────┘
            │
            │ execute()
            ▼
     ┌─────────────┐
     │   Notary    │
     │  validate   │
     └──────┬──────┘
            │
      ┌─────┴─────┐
      ▼           ▼
   VERIFIED     VOIDED
      │           │
      ▼           ▼
   commit()    discard
      │           │
      ▼           ▼
Model(v1)    Model(v0) unchanged
```