There are three distinct concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   CLUSTER (infrastructure)                                                  │
│   "What machines exist and how do they connect?"                            │
│                                                                             │
│   - Node discovery (mDNS, static config, cloud API)                         │
│   - Node registration ("I'm here, I have 4× A100")                          │
│   - Health monitoring (heartbeats, liveness)                                │
│   - Network topology (how do I reach node X?)                               │
│                                                                             │
│   This is the PLUMBING. It doesn't know about experiments or models.        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ provides connectivity
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ACTORS (logical entities)                                                 │
│   "What work needs to be done?"                                             │
│                                                                             │
│   - Notary, Experiment, Model, Storage, Backend                             │
│   - Can be instantiated on ANY node in the cluster                          │
│   - Communicate via Cap'n Proto messages                                    │
│   - Don't care about physical location                                      │
│                                                                             │
│   This is the LOGIC. Pure message-passing, location-transparent.            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ needs placement decisions
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ORCHESTRATOR (scheduler/router)                                           │
│   "Where should work happen?"                                               │
│                                                                             │
│   - "This protocol needs 80GB VRAM → route to vast.ai A100"                 │
│   - "This is a quick eval → run locally on RTX 4090"                        │
│   - Resource matching, job queuing, load balancing                          │
│   - Lease management (acquire/release compute)                              │
│                                                                             │
│   This is the DECISION LAYER. Knows cluster state + job requirements.       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

So the structure becomes:

```
caramba/
├── cluster/                         # Infrastructure: tying machines together
│   ├── node/
│   │   ├── node.capnp              # Node identity, capabilities
│   │   └── registry.py             # Track known nodes
│   ├── discovery/
│   │   ├── mdns.py                 # Local network discovery
│   │   ├── static.py               # Config file based
│   │   └── cloud.py                # AWS/GCP/Vast.ai API
│   ├── health/
│   │   ├── heartbeat.py
│   │   └── liveness.py
│   └── transport/
│       └── connection.py           # Cap'n Proto connection management
│
├── orchestrator/                    # Decision layer: where should work run?
│   ├── orchestrator.capnp          # Interface
│   ├── scheduler.py                # Job → Node matching
│   ├── router.py                   # Route requests to appropriate backend
│   └── lease.py                    # Acquire/release compute resources
│
├── actor/                           # Logical entities: what work to do
│   ├── notary/
│   ├── experiment/
│   ├── model/
│   ├── storage/
│   └── backend/
│       ├── backend.capnp           # Abstract interface
│       ├── torch/                  # PyTorch implementation
│       └── mlx/                    # MLX implementation
│
├── schema/                          # Shared data definitions
├── operation/                       # nn.Module implementations
├── compiler/                        # Manifest → Program pipeline
├── training/                        # Optimizer, scheduler, etc.
└── cli/
```

The flow when you run an experiment:

```
┌──────────────┐
│ caramba run  │
│ manifest.yml │
└──────┬───────┘
       │
       ▼
┌──────────────┐     "validate this manifest"      ┌──────────────┐
│  EXPERIMENT  │◀─────────────────────────────────▶│    NOTARY    │
│   (actor)    │                                   │   (actor)    │
└──────┬───────┘                                   └──────────────┘
       │
       │ "I need to run 12 protocols"
       │
       ▼
┌──────────────┐     "where should these run?"     ┌──────────────┐
│ ORCHESTRATOR │◀─────────────────────────────────▶│   CLUSTER    │
│              │                                   │              │
└──────┬───────┘                                   │ Node A: local│
       │                                           │   RTX 4090   │
       │ decisions:                                │              │
       │ - protocols 1-4 → local                   │ Node B: vast │
       │ - protocols 5-12 → vast.ai                │   8× A100    │
       │                                           └──────────────┘
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌──────────────┐                      ┌──────────────┐         │
│  │   BACKEND    │                      │   BACKEND    │         │
│  │   (local)    │                      │  (vast.ai)   │         │
│  │              │                      │              │         │
│  │ protocols    │                      │ protocols    │         │
│  │ 1, 2, 3, 4   │                      │ 5-12         │         │
│  └──────┬───────┘                      └──────┬───────┘         │
│         │                                     │                 │
│         │         ┌──────────────┐            │                 │
│         └────────▶│   STORAGE    │◀───────────┘                 │
│                   │    (S3)      │                              │
│                   │              │                              │
│                   │ checkpoints  │                              │
│                   │ metrics      │                              │
│                   └──────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The key insight is:

- **Cluster** knows about machines and connectivity
- **Orchestrator** knows about requirements and makes placement decisions  
- **Actors** just do their job, unaware of where they're running
