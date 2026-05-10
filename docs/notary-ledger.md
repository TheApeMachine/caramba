# Notary + Ledger (Lazy Invalidation)

Caramba needs a reliable definition of “what is true” at any given moment.
This document locks down the model:

- **Append-only ledger** records claims and evidence.
- A single **Notary** answers questions about truth **on demand**.
- **Lazy invalidation** is mandatory: we never pre-compute or eagerly propagate future invalidations.

This is designed to support research workflows where artifacts have deep dependency graphs, where
reproducibility matters, and where “truth” must be defensible.

## Core principle: nobody can assert global truth except the Notary

Any subsystem (trainer, evaluator, dataset pipeline, agent orchestrator, scheduler) can:

- emit events
- produce artifacts
- attach attestations (“I observed X under procedure P”)

But **only the Notary** is allowed to answer:

- “Is artifact A valid?”
- “Is run R reproducible?”
- “What is the current state of the project?”

This avoids the trap of distributed components silently making assumptions about global state.

## Why lazy invalidation (bank analogy)

If a teller counts money and records “there are exactly \(N\) bills”, that claim is only fully defensible up to the moment custody transfers.
After custody transfers, the teller cannot continue to assert the vault’s contents as truth.

Therefore:

- We **never** record “and therefore all downstream artifacts are invalid now”.
- We only record what we actually observed and what we can prove.
- When a user asks for current truth, the Notary re-validates **right now**, with the ledger as evidence.

Lazy invalidation is the only approach that avoids “recording assumptions about the future”.

## Ledger model

The ledger is an **append-only event log**. Entries are immutable.

Each entry should be thought of as one of:

- **Intent**: a declared plan (“I intend to run this manifest / benchmark / evaluation”)
- **Event**: a cause/effect record (“step finished”, “weights written”, “node joined”, “job leased”)
- **Claim**: a statement asserted under a procedure (“checksum matches”, “metric computed”)
- **ArtifactRef**: a pointer to immutable bytes (“checkpoint at content hash H”)

The ledger is not “the truth”. It is **evidence** from which truth can be derived.

## Truth model: “claims with proofs”

“True” in Caramba means:

> The Notary can produce a proof, at query time, that a claim/artifact is valid under the ledger up to some boundary.

That boundary is usually:

- **HEAD**: all entries known so far
- or **as-of time \(t\)** / **as-of entry id**: to reproduce historical views

Truth is therefore:

- **time-relative**
- **query-relative**

## What must be recorded so truth can be re-validated

Lazy validation only works if entries contain enough information to re-check invariants.

Every claim/artifact must include:

- **Identity**
  - Stable id (prefer content-addressed identifiers where possible)
  - Timestamp, author/system identity (who asserted this)
- **Dependencies**
  - Explicit `dependsOn` edges (other claim/artifact ids)
  - No implicit dependencies
- **Procedure / verification recipe**
  - “How was this produced?” (e.g., hash algorithm, environment lock, manifest hash, code revision)
  - Any parameters needed to reproduce the verification
- **Attestation**
  - A statement of what was observed and under what assumptions

If we cannot describe how to validate something, it is not a valid claim—at best it is an untrusted note.

## Notary responsibilities

The Notary:

- **Evaluates truth on demand**
  - walk the dependency DAG backward from the target claim/artifact
  - verify each dependency’s validity (recursively)
  - verify local invariants (hashes, manifest identity, environment lock, etc.)
- **Explains results**
  - if invalid, provide a minimal reason chain (“A depends on B; B cannot be validated because …”)
- **Caches safely**
  - caching is allowed, but cache entries are only valid relative to a specific ledger boundary
  - caches must be invalidated by new ledger entries (because new evidence can change provability)

Important: caching is not “eager invalidation”. It’s memoizing proofs that are valid **as-of a ledger snapshot**.

## Determination vs Outcome

We separate:

- **Determination**: verification work (“I checked these invariants and got these results”)
- **Outcome**: a notarial decision (“Given current ledger, claim C is provable / not provable”)

In an append-only system, “Outcome” does not delete history; it creates a new ledger entry describing the Notary’s conclusion.

## Practical implications for system design

- **Subsystems emit evidence, not truth**
  - they produce artifacts and attestations
  - they do not propagate invalidations
- **All dependencies must be explicit**
  - if a derived artifact depends on a dataset version, record the dataset identity
  - if it depends on a code revision, record the revision hash
- **The user gets a consistent answer**
  - “What is true right now?” is always answered by the same mechanism, not by scattered heuristics

## Interaction with Manifest (Driver) and Model (Collector)

- The **Manifest** drives intents and plans (what should happen).
- The **Model** collects events/artifacts (what happened).
- The **Notary** bridges the two:
  - “Does this record actually satisfy the plan?”
  - “Can we prove this artifact is a faithful output of that intent?”

This keeps Caramba reproducible and auditable while still being fast to iterate.

