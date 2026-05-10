# Agents as Ingress (the “Research Team”)

Caramba’s goal is to remove the precision/overhead burden of research while increasing insight and iteration speed.
That implies a third core concept beyond **Manifest (Driver)** and **Model (Collector)**:

- **Agents are the ingress (interaction surface)**: the place where ideas enter the system, get refined socially, and get converted into actionable plans.

This doc locks down how the “research team” agent system fits into Caramba’s architecture.

## The problem Agents solve

Research work is often:

- **Lonely** (lack of discussion and critique)
- **High-overhead** (experiments, provisioning, artifacts, reproducibility)
- **Brittle** (mistakes discovered late; backtracking is costly)

Agents address the “loneliness + idea ingestion” part by providing a structured, collaborative interface that can also **activate internal systems**.

## The “Research Team” interface

The frontend presents a single conversation surface that can route to major LLM providers—**OpenAI, Anthropic, and Google**—as provider-backed personas.

The key requirement is **non-linear group discussion**:

- Not just “human ↔ persona”
- But “persona ↔ persona” critique, reactions, and synthesis
- With the human able to steer, interrupt, and lock decisions

In other words: a *team*, not a chatbot.

## How Agents connect to Manifest/Model

Agents should not become “secret configuration.”

Instead, they operate as a **proposal engine**:

- **Agent output is a proposal**: a manifest patch, a run plan, an analysis, or a task request.
- **Execution is separate**: once accepted, Caramba executes via internal subsystems.
- **Everything is recorded**: proposals, accept/reject decisions, executions, artifacts, and evaluations go to the timeline.

This keeps the single-source-of-truth split intact:

- Manifest remains the upstream contract (driver)
- Model remains the downstream record (collector)
- Agents are the conversational *interface* that helps produce/choose those contracts

## Core architectural shape

Think of the agent system as three layers:

### 1) Provider adapters (LLM backends)

Each provider implements the same minimal capability:

- send message
- stream tokens/events
- return structured outputs (when supported)

This is where “Big 3” integration lives.

### 2) Personas (behavioral contracts)

Personas are configuration objects:

- role + constraints
- style + objectives
- domain knowledge emphasis

They must be explicit and versioned (so conversations are reproducible).

### 3) Orchestrator (team conversation dynamics)

The orchestrator is responsible for:

- turn-taking / parallel prompts
- cross-critique patterns (e.g. “critic reviews planner’s output”)
- synthesis / consensus building
- escalation (ask for more info, request experiments, run checks)

It decides *who speaks when* and *who responds to whom*.

## Interaction outputs (what Agents are allowed to do)

Agents can produce:

- **Text discussion** (human-readable reasoning and critique)
- **Proposals** (structured):
  - manifest patches
  - experiment/run requests
  - evaluation/benchmark requests
  - refactors that improve reproducibility
- **Delegations**:
  - “run this benchmark on available nodes”
  - “validate this manifest”
  - “compare these two runs”

Crucially:

- Agents may *suggest* actions.
- The system must make actions **explicit**, **auditable**, and **recorded**.

## Contract-first implementation (Cap’n Proto-friendly)

To keep this “real” and distributed-friendly:

- Conversations, messages, proposals, and task requests should be typed messages (Cap’n Proto structs).
- Agent execution should be capabilities (interfaces) where it makes sense:
  - a “Chat” capability
  - a “TaskDispatcher” capability
  - a “Timeline” capability for writing events

This keeps the agent layer composable and makes it easy to distribute:

- local-only: everything runs on one machine
- cluster: orchestrator runs locally, providers run elsewhere, or vice versa

## Non-negotiable rules

- **No hidden authority**: agent outputs never silently change manifests or run configs.
- **Every action is explicit**: proposals must be accepted/rejected; execution must be recorded.
- **Reproducibility includes the conversation**: persona versions and prompts are part of the run context.

