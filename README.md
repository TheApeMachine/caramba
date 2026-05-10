# caramba

**A substrate for AI research**

---

## The Problem

In a typical ML workflow, state is scattered: configs live in YAML files, metrics in W&B, checkpoints on S3, decisions in Slack threads. Each handoff between systems is a point where certainty is lost.

Consider a bank counter who counts money in the vault. They know the count exactly—until they hand the money to a security guard. From that moment, they can no longer vouch for the count with absolute certainty. They didn't oversee the full chain of custody.

This is the state of ML research today. You run an experiment, and somewhere between your config file, your training script, your checkpoint, and your results table, you lose the ability to say with certainty: *"I know exactly how this model was produced."*

Caramba solves this by never letting go of the money.

---

## The Insight

**Everything accumulates forward.**

Instead of scattering state across systems, Caramba chains contracts where each accumulates into the next:

```
MANIFEST (driver) → PROTOCOL (actor) → MODEL (collector)
```

| Contract     | Role                             | Contains                                                   |
|--------------|----------------------------------|------------------------------------------------------------|
| **Manifest** | Declares the researcher's intent | What we're trying to learn                                 |
| **Protocol** | Defines procedures for execution | How to execute, embeds the Manifest                        |
| **Model**    | Outcome *and* ledger             | Architecture, weights, full audit trail, embeds everything |

A **`.cbm` file** is the **Caramba model bundle**: the serialized weights plus manifests, protocol references, ledger slices, and other metadata/config needed to load, inspect (`caramba inspect model.cbm`), or share a Model as one artifact.

The Model isn't just the output—it's the complete provenance. When you share a Model, you share its entire history. When you resume training, you resume from complete state, not a partial snapshot.

→ [Deep dive: Governance Model](./docs/governance.md)

---

## The Notary

The Notary maintains continuous custody over the entire research process. It is the single source of truth—the one system that never loses sight of the money.

```
┌────────────────────────────────────────────────────────────────┐
│                           NOTARY                               │
│                   (single source of truth)                     │
├────────────────────────────────────────────────────────────────┤
│             MANIFEST ──→ PROTOCOL ──→ MODEL                    │
│             (driver)     (actor)   (collector)                 │
└────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                   RESEARCH PROJECT                             │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌─────────────────────────┐    ┌───────────┐  │
│  │ PROTOCOL │───▶│ EXPERIMENT              │───▶│   MODEL   │  │
│  └──────────┘    │ deploy → train → bench  │    └───────────┘  │
│                  └─────────────────────────┘          │        │
│                             │                         │        │
│                             ▼                         ▼        │
│                       ┌──────────┐             ┌──────────┐    │
│                       │ APPROVED │────────────▶│ VERIFIED │    │
│                       └──────────┘             └──────────┘    │
│                                                      │         │
│                                                      ▼         │
│                             ┌─────────────────────────────┐    │
│                             │        VALIDATION           │    │
│                             │  pass: commit to new truth  │    │
│                             │  fail: void, rollback       │    │
│                             └─────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Submit** — Researcher submits a Manifest declaring intent
2. **Validate** — Notary checks compatibility with Protocols and current Model state
3. **Execute** — Experiment runs on a copy of the Model; weights are updated in the copy only (the original model is not modified).
4. **Checkpoint** — At defined points, Notary validates against Protocol expectations
5. **Commit or Void** — Pass: the copy becomes the new source of truth. Fail: the copy is destroyed, original remains untouched.

→ [Deep dive: The Notary](./docs/notary.md)

---

## Atomic Intent

A Manifest declares a unit of scientific intent. Not a batch of runs—an *intent*.

```yaml
# This is ONE intent, not twelve independent experiments
name: dba_ablation_study
variants: [baseline, bottleneck, decoupled, gqa]
seeds: [1337, 1338, 1339]
```

If you declare "compare architectures A, B, C, D across seeds 1, 2, 3," that comparison is atomic. If `bottleneck_s1338` fails, you don't have eleven good runs and one bad one. You have zero complete ablation studies.

### Why All-or-Nothing?

Consider the alternative:

1. You run 12 experiments
2. 3 fail due to a config bug
3. You comment them out, re-run just those 3
4. Now you have results from two sessions, possibly different code versions
5. You manually track which belong together
6. Six months later: *"wait, were these all from the same sweep?"*

Caramba prevents this by design. A quick void and clean re-run is nearly always cheaper than discovering later that your published comparison had a confound you forgot about.

**Recovery workflow (forward pointer):** when an intent voids, use the preserved attempt record ([Voiding Isn't Waste](#voiding-isnt-waste) below, plus [Notary & ledger](./docs/notary-ledger.md) for auditing) to inspect failures, adjust configs, then resubmit the **entire** atomic comparison—never a cherry-picked subset. Intermediate timelines, checkpoints, ledger entries, and failure diagnostics remain available until you explicitly supersede them with a verified run ([Validation & Trust](./docs/validation.md) lays out checkpoints and audit expectations).

**There is no "approved" with an asterisk.**

→ [Deep dive: Atomic Intent](./docs/atomic-intent.md)

---

## Trust by Design

When a Model carries approval from the Notary, that approval means something:

- ✓ Every run in the Manifest completed
- ✓ Every validation checkpoint passed  
- ✓ Full provenance is intact and auditable
- ✓ No partial results, no manual stitching, no "I think this was from the same sweep"

The system doesn't allow ambiguity to exist.

### Voiding Isn't Waste

Voiding an Experiment doesn't mean compute is lost. The Notary records what was attempted, where it failed, and the state at failure. You can examine the voided Experiment, fix the issue, and resubmit. But you cannot accidentally ship results from a half-completed study.

→ [Deep dive: Validation & Trust](./docs/validation.md)

---

## Quick Start

```bash
# Install
pip install caramba

# Run a manifest
caramba run manifest.yaml

# Check status
caramba status

# Inspect a model's provenance
caramba inspect model.cbm
```

→ [Getting Started Guide](./docs/getting-started.md)

---

## Documentation

| Document                                           | Description                                    |
|----------------------------------------------------|------------------------------------------------|
| [Getting Started](./docs/getting-started.md)       | Installation, first experiment, basic workflow |
| [Governance Model](./docs/governance.md)           | Manifest → Protocol → Model in depth           |
| [The Notary](./docs/notary.md)                     | Custody, validation, the approval process      |
| [Atomic Intent](./docs/atomic-intent.md)           | Why partial results don't exist                |
| [Validation & Trust](./docs/validation.md)         | Checkpoints, void conditions, audit trails     |
| [Manifest Reference](./docs/manifest-reference.md) | Complete YAML schema and examples              |
| [Protocol Library](./docs/protocols.md)            | Built-in protocols and how to extend them      |
| [Architecture](./docs/architecture.md)             | System internals for contributors              |

---

## License

MIT

---

<p align="center">
  <i>The model knows how it was made.</i>
</p>