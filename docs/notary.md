# The Notary

---

## Core Principle

Nobody can assert global truth except the Notary.

Any subsystem—trainer, evaluator, dataset pipeline, agent orchestrator, scheduler—can emit events, produce artifacts, and attach attestations ("I observed X under procedure P"). But only the Notary is allowed to answer:

- "Is artifact A valid?"
- "Is run R reproducible?"
- "What is the current state of the project?"

This avoids the trap of distributed components silently making assumptions about global state.

---

## The Bank Analogy

If a teller counts money and records "there are exactly N bills," that claim is defensible only up to the moment custody transfers. After custody transfers, the teller cannot continue to assert the vault's contents as truth.

Therefore:

- We **never** record "and therefore all downstream artifacts are invalid now."
- We only record what we actually observed and what we can prove.
- When a user asks for current truth, the Notary re-validates **right now**, using the ledger as evidence.

Lazy invalidation is the only approach that avoids "recording assumptions about the future."

---

## The Ledger

The ledger is an append-only event log. Entries are immutable. Each entry is one of:

| Entry type    | Description                                                              |
|---------------|--------------------------------------------------------------------------|
| **Intent**    | A declared plan ("I intend to run this manifest / benchmark / eval")     |
| **Event**     | A cause/effect record ("step finished", "weights written", "job leased") |
| **Claim**     | A statement asserted under a procedure ("checksum matches", "metric computed") |
| **ArtifactRef** | A pointer to immutable bytes ("checkpoint at content hash H")          |

The ledger is not the truth. It is **evidence** from which truth can be derived.

### What Every Entry Must Contain

Lazy validation only works if entries contain enough information to re-check invariants:

| Field               | Purpose                                                           |
|---------------------|-------------------------------------------------------------------|
| **Identity**        | Stable ID, timestamp, author/system identity                     |
| **Dependencies**    | Explicit `dependsOn` edges to other claim/artifact IDs           |
| **Procedure**       | How was this produced? Hash algorithm, manifest hash, code revision |
| **Attestation**     | What was observed, under what assumptions                        |

If we cannot describe how to validate something, it is not a valid claim—at best it is an untrusted note.

---

## Truth Model

"True" in Caramba means:

> The Notary can produce a proof, at query time, that a claim or artifact is valid under the ledger up to some boundary.

That boundary is:

- **HEAD** — all entries known so far
- **as-of time t** — to reproduce a historical view

Truth is therefore **time-relative** and **query-relative**.

---

## Notary Responsibilities

### Evaluate truth on demand

1. Walk the dependency DAG backward from the target claim/artifact
2. Verify each dependency's validity (recursively)
3. Verify local invariants (hashes, manifest identity, environment lock)

### Explain results

If invalid, provide a minimal reason chain: "A depends on B; B cannot be validated because…"

### Cache safely

Caching uses snapshot-based memoization: a proof is keyed to a ledger **snapshot identifier** rather than invalidated eagerly on each append.

When the ledger advances, older cache entries remain valid only relative to older snapshots. Answers for "truth right now" must be recomputed or served from proofs tied to newer snapshots.

Caches never pretend to prove something "globally"—they summarize verification work for `snapshot = S` until that snapshot moves forward.

---

## Determination vs Outcome

These are separate concepts:

**Determination** — verification work performed:
> "Verified checksum of artifact A matches hash H under manifest M."

**Outcome** — a notarial decision:
> "Ledger entry #1234 (HEAD): Notary confirms artifact A is admissible for claim C."

In an append-only system, "Outcome" does not delete history. It creates a new ledger entry describing the Notary's conclusion.

---

## Validation Flow

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

### Validation Steps

1. **Submit** — Researcher submits a Manifest
2. **Validate** — Notary checks manifest against Protocols and current Model state
3. **Execute** — Experiment runs on a copy of the Model
4. **Checkpoint** — At defined intervals, Notary validates against Protocol expectations
5. **Commit or Void** — Pass: copy becomes new Model. Fail: copy destroyed, original untouched.

---

## Voiding Is Not Waste

When an experiment is voided, the Notary records what was attempted, where it failed, and the state at failure. You can examine the voided Experiment, fix the issue, and resubmit.

What you cannot do is accidentally ship results from a half-completed study.

The voided attempt record is preserved in the ledger, including:
- Which step or checkpoint triggered the void
- What invariant was violated
- The state of all artifacts at the time of voiding

---

## Interaction with Manifest and Model

- The **Manifest** drives intents and plans (what should happen)
- The **Model** collects events and artifacts (what happened)
- The **Notary** bridges the two:
  - "Does this record actually satisfy the plan?"
  - "Can we prove this artifact is a faithful output of that intent?"

---

## Design Rules

- **Subsystems emit evidence, not truth.** They produce artifacts and attestations. They do not propagate invalidations.
- **All dependencies must be explicit.** If a derived artifact depends on a dataset version, record the dataset identity. If it depends on a code revision, record the revision hash.
- **The user gets a consistent answer.** "What is true right now?" is always answered by the same mechanism, not by scattered heuristics.
