# Manifest (Driver) and Model (Collector)

Caramba’s architecture is easiest to reason about if we lock down two “single sources of truth”:

- **The Manifest is the driver (upstream source of truth)**: it describes *what should happen*.
- **The Model is the collector (downstream source of truth)**: it records *what did happen*.

This framing is intentionally simple. It’s a guardrail against configuration drift, hidden defaults,
and “just one more CLI flag” syndrome.

## Manifest = Driver (what should happen)

The manifest is the **immutable contract** for a research project and its runs.

In practice this means:

- **All decisions are derived from the manifest** (and its resolved/compiled forms), not from ad-hoc flags.
- **No silent fallbacks**: if the manifest doesn’t specify it (directly or via a resolved variable), it doesn’t exist.
- The manifest is the “plan” for the run:
  - architecture/topology
  - datasets and transforms
  - training schedule
  - evaluation and deployment targets
  - resource constraints and execution target selection

## Model = Collector (what did happen)

The model is a **stateful sink for outcomes** produced by executing the plan.

In practice this means:

- The model is where we attach *observations* and *results*:
  - step counters and phase state
  - logits/loss/metrics
  - checkpoints and artifacts
  - traces, timings, and provenance
- The model is the “record” for the run:
  - an append-only timeline of events
  - pointers to immutable artifacts (content-addressed where possible)

## Concrete example: `PhasedCycler`

`framework/training/cycler/phase.py` already follows this pattern:

- **Driver**: `manifest`, `target`, and `run` determine the program, backend, optimizer, system, and dataset.
- **Collector**: `Model` accumulates `step`, `logits`, and `loss` and is the natural place to attach events/artifacts.

The cycler’s job is orchestration: apply the plan (manifest) and populate the record (model).

## The boundary we care about: Plan vs Record

If you want a crisp contract boundary:

- **Run Plan**: the resolved/compiled manifest-derived “intent”
  - deterministic inputs
  - reproducible configuration
  - stable graph/program description
- **Run Record**: the emitted history of execution
  - event stream (timeline)
  - artifacts (checkpoints, eval reports, plots)
  - derived metrics and diagnostics

This is the seam where Cap’n Proto becomes extremely valuable—**Cap’n Proto** (“Cap’n Proto is a fast data interchange format and RPC system”; see [`https://capnproto.org/`](https://capnproto.org/)):

- Plans and records can be defined as **Cap’n Proto structs**.
- Components can communicate via **typed messages** and/or **capabilities** while keeping the same conceptual split.

## Design rules (non-negotiable)

- **Manifest is upstream truth**: no environment-variable config, no CLI flags for core behavior.
- **Model is downstream truth**: results come from execution; don’t “recompute history” from logs.
- **Fail hard and fast**: if plan/record invariants are violated, raise immediately with a clear error.

