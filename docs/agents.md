# Agents

---

## Agents as Ingress

Agents serve as the **conversational ingress**—the place where ideas enter the system, get refined through structured discussion, and get converted into actionable plans.

Three concerns define research work today:

- **Lonely** — lack of discussion and critique
- **High-overhead** — experiments, provisioning, artifacts, reproducibility
- **Brittle** — mistakes discovered late; backtracking is costly

Agents address the first by providing a structured, multi-persona interface that can also activate internal systems.

---

## The Research Team

The frontend presents a single conversation surface that routes to major LLM providers (OpenAI, Anthropic, Google) as provider-backed personas. The key requirement is **non-linear group discussion**:

- Not just human-to-persona exchange
- But persona-to-persona critique, reactions, and synthesis
- With the human able to steer, interrupt, and lock decisions

This is a *team*, not a chatbot.

---

## How Agents Connect to the System

Agents are a **proposal engine**. They never silently modify state.

```
AGENT OUTPUT (proposal):
  - Manifest patch
  - Run plan or architecture change
  - Analysis or critique
  - Task request ("validate this manifest", "compare these runs")

         ▼ requires explicit human accept/reject

EXECUTION (internal systems):
  - Manifest submitted to Notary
  - Experiment launched and recorded
  - Results appended to ledger
```

The single-source-of-truth split remains intact:
- **Manifest** remains the upstream contract (driver)
- **Model** remains the downstream record (collector)
- **Agents** are the conversational interface that helps produce and choose those contracts

---

## Three Layers

### 1. Provider Adapters

Each LLM provider implements the same minimal capability:
- Send message
- Stream tokens/events
- Return structured outputs (tool calls, JSON mode)

This is where OpenAI, Anthropic, and Google integration lives.

### 2. Personas

Personas are explicit, versioned configuration objects:

```yaml
name: critic
role: |
  Identify methodological flaws, confounds, and threats to validity.
  Do not suggest alternatives—raise problems only. Be concise and specific.
constraints:
  - Do not propose solutions; only identify issues
  - Cite specific parts of the manifest or design
  - Limit responses to 3 bullet points
style: adversarial
domain_emphasis:
  - experimental design
  - statistical validity
  - reproducibility
```

Personas are versioned because conversations must be reproducible. The persona version is part of the run context recorded in the ledger.

### 3. Orchestrator

The orchestrator manages team dynamics:

- **Turn-taking** — who speaks when
- **Cross-critique** — critic reviews planner's output; synthesizer integrates feedback
- **Parallel prompts** — multiple personas respond to the same input simultaneously
- **Synthesis** — aggregating multiple perspectives into a coherent proposal
- **Escalation** — requesting more information, proposing experiments, running checks

---

## What Agents Can Produce

| Output type     | Description                                              |
|-----------------|----------------------------------------------------------|
| Text discussion | Human-readable reasoning and critique                    |
| Manifest patch  | A proposed change to the research manifest               |
| Run request     | A proposed experiment or benchmark to execute            |
| Delegation      | "Run this benchmark on available nodes"                  |
| Analysis        | Post-hoc interpretation of results, checkpoint diffs     |

Every output that could cause a system action is a **proposal**. The human must explicitly accept it before execution. Everything is recorded.

---

## Developer Sub-Agents

The developer agent has a `sub_agent` tool for short-lived parallel delegation. Each call submits one or more tasks through `qpool`; every task gets its own provider conversation, its own system prompt, and its own user prompt.

Sub-agents are intentionally read-only. They can search code and view files, but they cannot edit files, create files, run shell commands, or signal task completion. Their output is returned to the parent developer agent in the same order the tasks were requested, so the parent keeps responsibility for synthesis and any code changes.

Each `sub_agent` call also creates a qpool broadcast group shared by the sibling agents in that call. Sub-agents can use `publish_finding` to share discoveries, `read_peer_findings` to drain messages from other agents, and `list_peers` to see who is subscribed. This keeps coordination inside qpool while preserving isolated LLM context windows.

Tool payload shape:

```json
{
  "tasks": [
    {
      "name": "api-reader",
      "system_prompt": "You map package boundaries and report only facts.",
      "user_prompt": "Find the provider-neutral tool interfaces and summarize how tool calls are dispatched."
    }
  ]
}
```

---

## Non-Negotiable Rules

- **No hidden authority.** Agent outputs never silently change manifests or run configs.
- **Every action is explicit.** Proposals must be accepted or rejected; execution must be recorded.
- **Reproducibility includes the conversation.** Persona versions and prompts are part of the run context.
