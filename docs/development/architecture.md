Cap'n Proto isn't just serialization—it's the **message-passing substrate** for a distributed actor system. Each component is an independent entity that can live anywhere, communicating via typed messages.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         DISTRIBUTED ACTOR MODEL                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│    ┌──────────┐          messages           ┌──────────┐                   │
│    │          │◀───────────────────────────▶│          │                   │
│    │  NOTARY  │        (Cap'n Proto)        │ STORAGE  │                   │
│    │          │                             │          │                   │
│    └────▲─────┘                             └──────────┘                   │
│         │                                         ▲                        │
│         │ messages                                │                        │
│         │                                         │                        │
│    ┌────┴─────┐          messages           ┌─────┴────┐                   │
│    │          │◀───────────────────────────▶│          │                   │
│    │EXPERIMENT│        (Cap'n Proto)        │  MODEL   │                   │
│    │          │                             │          │                   │
│    └────▲─────┘                             └──────────┘                   │
│         │                                                                  │
│         │ messages                                                         │
│         │                                                                  │
│    ┌────┴─────────────────────────────────────────────────────┐            │
│    │                                                          │            │
│    │                      BACKENDS                            │            │
│    │                                                          │            │
│    │  ┌──────────┐    ┌──────────┐    ┌──────────┐            │            │
│    │  │  LOCAL   │    │  VAST.AI │    │    GCP   │            │            │
│    │  │ RTX 4090 │    │  A100×8  │    │  TPU v4  │            │            │
│    │  └──────────┘    └──────────┘    └──────────┘            │            │
│    │                                                          │            │
│    └──────────────────────────────────────────────────────────┘            │
│                                                                            │
│    Each box can be a separate process, machine, or cloud instance.         │
│    All communication is typed messages over Cap'n Proto RPC.               │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

In Alan Kay's vision, objects are like biological cells or networked computers—they have their own state, they receive messages, they decide how to respond. The *interface* is the message protocol, not the internal implementation.

So in Cap'n Proto terms:

```capnp
# notary.capnp

interface Notary {
  # Validate a manifest against current model state
  validateManifest @0 (manifest :Manifest, model :Model) -> (result :ValidationResult);
  
  # Validate a checkpoint during execution
  validateCheckpoint @1 (checkpoint :Checkpoint, expected :Expected) -> (result :ValidationResult);
  
  # Final validation before commit
  validateFinal @2 (experiment :Experiment) -> (result :ValidationResult);
  
  # Query truth from ledger
  audit @3 (claim :Text, asOf :UInt64) -> (result :AuditResult);
}

interface Experiment {
  # Get current state
  getState @0 () -> (state :ExperimentState);
  
  # Execute a protocol (might dispatch to remote backend)
  executeProtocol @1 (protocol :Protocol) -> (run :Run);
  
  # Checkpoint for validation
  checkpoint @2 () -> (checkpoint :Checkpoint);
  
  # Commit to model (only after Notary approval)
  commit @3 (approval :Approval) -> (model :Model);
  
  # Void and discard
  void @4 (reason :Text) -> ();
}

interface Backend {
  # Compile architecture to executable form
  compile @0 (architecture :Architecture) -> (program :Program);
  
  # Execute training step
  step @1 (program :Program, batch :Batch) -> (result :StepResult);
  
  # Get capabilities (what can this backend do?)
  capabilities @2 () -> (caps :BackendCapabilities);
}
```

The powerful thing about Cap'n Proto interfaces is **capability passing**. When you call `executeProtocol`, you get back a `Run` *capability*—a live reference to a running process that might be on a different machine:

```capnp
interface Run {
  # Query current state
  getMetrics @0 () -> (metrics :Metrics);
  getStep @1 () -> (step :UInt64);
  
  # Stream events as they happen
  subscribe @2 (subscriber :RunSubscriber) -> ();
  
  # Control
  pause @3 () -> ();
  resume @4 () -> ();
  cancel @5 () -> ();
}

interface RunSubscriber {
  # Called by Run when events occur
  onStep @0 (event :StepEvent) -> ();
  onCheckpoint @1 (checkpoint :Checkpoint) -> ();
  onComplete @2 (result :RunResult) -> ();
  onError @3 (error :Error) -> ();
}
```

This means the Experiment can dispatch training to a Vast.ai A100 cluster, and get back a capability that lets it monitor progress, receive events, and control execution—all through message passing, regardless of where the compute actually runs.

Here's how the distributed flow might look:

```
┌─────────────────┐
│   RESEARCHER    │
│   (laptop)      │
└────────┬────────┘
         │
         │ caramba run manifest.yml
         ▼
┌─────────────────┐         validateManifest()         ┌─────────────────┐
│   EXPERIMENT    │◀──────────────────────────────────▶│     NOTARY      │
│   (laptop)      │                                    │   (laptop or    │
│                 │                                    │    cloud)       │
└────────┬────────┘                                    └─────────────────┘
         │
         │ executeProtocol(baseline_s1337)
         │
         │ Backend.capabilities() → {A100: vast.ai, RTX4090: local}
         │ 
         │ "This job needs 40GB VRAM, dispatch to vast.ai"
         │
         ▼
┌─────────────────┐
│  BACKEND ROUTER │
│   (laptop)      │
└────────┬────────┘
         │
         │  spawn worker, pass Program capability
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│                         VAST.AI CLUSTER                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   WORKER    │  │   WORKER    │  │   WORKER    │             │
│  │ baseline    │  │ bottleneck  │  │ decoupled   │  ...        │
│  │ s1337       │  │ s1337       │  │ s1337       │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                    │
│         │ onStep()       │ onStep()       │ onStep()           │
│         │ onCheckpoint() │ onCheckpoint() │ onCheckpoint()     │
│         │                │                │                    │
└─────────┴────────────────┴────────────────┴────────────────────┘
          │                │                │
          │                │                │
          ▼                ▼                ▼
     ┌─────────────────────────────────────────┐
     │              EXPERIMENT                 │
     │  (aggregates events, coordinates)       │
     │                                         │
     │  validateCheckpoint() → Notary          │
     │  saveWeights() → Storage                │
     └─────────────────────────────────────────┘
```

The key insight is that **capabilities are first-class**. When you get a reference to a `Run`, you can:

1. Pass it to someone else (e.g., a monitoring dashboard)
2. Store it and use it later
3. Revoke it (cancel access)

This is very different from traditional RPC where you just call methods. Here, you're passing around *live references* to distributed objects.

For the storage side, it's similar:

```capnp
interface Storage {
  # Basic operations
  get @0 (path :Text) -> (data :Data);
  put @1 (path :Text, data :Data) -> ();
  exists @2 (path :Text) -> (exists :Bool);
  list @3 (prefix :Text) -> (paths :List(Text));
  
  # Streaming for large objects
  getStream @4 (path :Text) -> (stream :ReadStream);
  putStream @5 (path :Text) -> (stream :WriteStream);
  
  # Transactions (for atomic commits)
  beginTransaction @6 () -> (txn :Transaction);
}

interface Transaction {
  put @0 (path :Text, data :Data) -> ();
  delete @1 (path :Text) -> ();
  commit @2 () -> ();
  rollback @3 () -> ();
}
```

So the Model commit becomes an atomic transaction:

```python
async def commit(self, approval: Approval) -> Model:
    """Commit experiment to a new Model version."""
    
    txn = await self.storage.beginTransaction()
    
    try:
        # Write all artifacts atomically — serialize payloads off-thread/async so commits never block UI/workers while buffers are hydrated.
        for protocol_id, weights in self.weights.items():
            await txn.put(f"weights/{protocol_id}.safetensors", weights)
        
        for protocol_id, metrics in self.metrics.items():
            await txn.put(f"metrics/{protocol_id}.json", metrics)
        
        ledger_blob = await self.ledger.serialize()
        index_blob = await self.build_index().serialize()

        await txn.put("ledger/events.jsonl", ledger_blob)
        await txn.put("model.capnp", index_blob)
        
        await txn.commit()
        
    except Exception as e:
        await txn.rollback()
        raise
    
    return Model(storage=self.storage, index=self.build_index())
```

The system becomes a network of actors passing typed messages, with Cap'n Proto providing:

1. **Schema** — what data looks like
2. **Interfaces** — what messages objects accept  
3. **Capabilities** — live references that can be passed around
4. **Transport** — efficient serialization and RPC