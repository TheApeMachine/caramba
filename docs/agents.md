# 🤖 Agent Workflows

caramba includes AI-assisted research automation through agent processes. These agents can draft papers, review experiments, and run autonomous research loops.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Process Types](#process-types)
- [Paper Writing](#paper-writing)
- [Paper Review](#paper-review)
- [Research Loop](#research-loop)
- [Idle Loop](#idle-loop)
- [Knowledge Integration](#knowledge-integration)
- [Configuration](#configuration)

---

## Overview

Agent workflows are implemented as **process targets** in manifests:

```yaml
targets:
  - type: process
    name: paper_write
    team:
      writer: writer
    process:
      type: paper_write
      writer: writer
      output_dir: paper
```

These processes coordinate AI agents to:
- 📝 Draft academic papers from experiment results
- 🔍 Review papers and suggest improvements
- 🔄 Run write → review → experiment loops
- 🧠 Integrate knowledge from codebase and research

---

## Process Types

### Available Processes

| Process           | Purpose                               | Key Agents               |
|-------------------|---------------------------------------|--------------------------|
| `paper_write`     | Generate LaTeX paper from experiments | writer                   |
| `paper_review`    | Review paper and propose experiments  | reviewer                 |
| `research_loop`   | Write → Review → Audit loop           | leader, writer, reviewer |
| `idle`            | Budgeted readiness + eval + research loop | leader, writer, reviewer |
| `discussion`      | Multi-agent research discussion       | multiple researchers     |
| `code_graph_sync` | Index codebase into knowledge graph   | -                        |
| `paper_collect_artifacts` | Collect benchmark artifacts into paper-ready tables/figures | - |
| `platform_improve`| Ingest → ideate → implement → verify → PR (in Docker workspace) | leader, ideators, developer, verifier |
| `multiplex_chat`  | Interactive shared-context chat across multiple model providers | chatgpt, claude, gemini |

### Process Target Structure

```yaml
- type: process
  name: my_process
  team:
    role_name: persona_name
  process:
    type: process_type
    # Process-specific configuration
```

---

## Paper Writing

Generate a complete LaTeX paper from experiment results.

### Basic Configuration

```yaml
targets:
  - type: process
    name: paper_write
    team:
      writer: writer
    process:
      type: paper_write
      name: paper_write
      writer: writer
      output_dir: paper
```

### What the Agent Does

1. **Reads experiment manifest** — Understands what was run
2. **Collects results** — Gathers benchmark data and artifacts
3. **Generates LaTeX** — Creates structured paper sections
4. **Includes figures** — References generated charts
5. **Manages bibliography** — Creates `references.bib`

### Output Structure

```text
artifacts/
└── experiment_name/
    └── paper/
        ├── paper.tex          # Main LaTeX document
        ├── references.bib     # Bibliography
        └── figures/           # Included figures
            ├── summary.png
            ├── latency_vs_context.png
            └── ...
```

### Paper Sections

The writer generates standard academic sections:

| Section      | Content                       |
|--------------|-------------------------------|
| Abstract     | Summary of contributions      |
| Introduction | Problem statement, motivation |
| Related Work | Prior research context        |
| Methodology  | Technical approach            |
| Experiments  | Setup, datasets, baselines    |
| Results      | Quantitative findings         |
| Discussion   | Analysis and limitations      |
| Conclusion   | Summary and future work       |

### Running Paper Writing

```bash
# Configure in manifest
python3 -m caramba manifest.yml

# Or run paper target specifically
python3 -m caramba manifest.yml --target paper_write
```

---

## Multiplex Chat (ChatGPT / Claude / Gemini)

An interactive REPL process that lets you route each turn to a specific model
using `@chatgpt`, `@claude`, and `@gemini`, while sharing one transcript context.

### Configuration

```yaml
targets:
  - type: process
    name: multiplex_chat
    team:
      chatgpt: chatgpt
      claude: claude
      gemini: gemini
    process:
      type: multiplex_chat
      name: multiplex_chat
      routes:
        chatgpt: chatgpt
        claude: claude
        gemini: gemini
      initial_route: chatgpt
      stream: false
```

### Running

```bash
python3 -m caramba config/presets/multiplex_chat.yml --target multiplex_chat
```

### Required API keys

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`

The default personas in `config/personas/chatgpt.yml`, `claude.yml`, and
`gemini.yml` use LiteLLM routes for Anthropic/Gemini.

---

## Paper Review

Review an existing paper and propose improvements or new experiments.

### Configuration

```yaml
targets:
  - type: process
    name: paper_review
    team:
      reviewer: reviewer
    process:
      type: paper_review
      name: paper_review
      reviewer: reviewer
      strictness: conference
      max_proposed_experiments: 3
      output_dir: paper
```

### Strictness Levels

| Level        | Focus                                   |
|--------------|-----------------------------------------|
| `workshop`   | Novel ideas, early-stage work           |
| `conference` | Complete evaluation, clear presentation |
| `journal`    | Comprehensive coverage, reproducibility |
| `top_venue`  | State-of-the-art, significant impact    |

### Reviewer Actions

| Action           | Description                     |
|------------------|---------------------------------|
| `approve`        | Paper is ready                  |
| `style_fix`      | Minor stylistic changes         |
| `clarification`  | Needs clarification             |
| `new_experiment` | Requires additional experiments |
| `major_revision` | Significant restructuring       |

### Reviewer Personas

| Persona         | Focus                                                     |
|-----------------|-----------------------------------------------------------|
| `reviewer`      | Critical, actionable review + Graphiti-based safety audit |
| `ml_expert`     | ML feasibility + eval/benchmark rigor                     |
| `mathematician` | Correctness, proofs, edge cases                           |
| `architect`     | Platform design + migration safety                         |

### Review Output

```json
{
  "overall_score": 7.5,
  "recommendation": "new_experiment",
  "strengths": [
    "Novel approach to attention compression",
    "Comprehensive benchmarks"
  ],
  "weaknesses": [
    "Missing ablation on bottleneck dimensions",
    "Limited baselines"
  ],
  "proposed_experiments": [
    {
      "name": "ablation_bottleneck",
      "hypothesis": "Larger bottleneck improves quality",
      "priority": "high"
    }
  ]
}
```

---

## Research Loop

Autonomous write → review → structural audit loop.

### Architecture

```text
┌─────────────────────────────────────────────────────┐
│                 RESEARCH LOOP                       │
├─────────────────────────────────────────────────────┤
│ ┌──────────┐    ┌──────────┐    ┌─────────────────┐ │
│ │  Write   │───▶│  Review  │───▶│ Structural      │ │
│ │  Paper   │    │  Paper   │    │ Audit           │ │
│ └────▲─────┘    └──────────┘    └────────┬────────┘ │
│      │                                   │          │
│      │          ┌──────────────┐         │          │
│      │          │   Address    │         │          │
│      └──────────│  Weaknesses  │◀────────┘          │
│                 └──────────────┘                    │
│ Repeat until: approved OR max iterations reached    │
└─────────────────────────────────────────────────────┘
```

### Configuration

```yaml
targets:
  - type: process
    name: research_loop
    team:
      leader: research_lead
      writer: writer
      reviewer: reviewer
    process:
      type: research_loop
      name: research_loop
      leader: leader
      writer: writer
      reviewer: reviewer
      max_iterations: 5
      auto_run_experiments: false
      output_dir: paper
```

### Loop Behavior

1. **Write** — Generate/update paper based on feedback
2. **Review** — Evaluate paper, identify weaknesses
3. **Structural Audit** — Query knowledge graph for safety
4. **Check Approval** — Stop if score ≥ threshold
5. **Address Weaknesses** — Incorporate feedback
6. **Repeat** — Until approved or max iterations

### Output

```text
artifacts/
└── experiment_name/
    └── agents/
        └── research_loop/
            ├── iteration_1_*.json
            ├── iteration_2_*.json
            └── ...
```

---

## Idle Loop

Budgeted background work: readiness + evaluation + a short research loop pass.

This is intended to be safe by default: it can sync structure, run cheap checks,
and write diffable artifacts, all within a strict wall-clock budget.

### Configuration

```yaml
targets:
  - type: process
    name: idle
    team:
      leader: research_lead
      writer: writer
      reviewer: reviewer
    process:
      type: idle
      name: idle
      max_wall_time_sec: 600

      # Readiness: index current model topology into Graphiti
      run_code_graph_sync: true
      code_graph_sync_agent: leader
      index_namespace: main

      # Continuous eval (optional): short deterministic commands
      run_eval: true
      eval_cmds:
        - python -m pytest -q
      eval_timeout_sec: 300
      eval_cwd: .

      # Research loop (optional): one quick iteration
      run_research_loop: true
      leader: leader
      writer: writer
      reviewer: reviewer
      research_max_iterations: 1
      research_auto_run_experiments: false
      output_dir: paper
```

---

## Knowledge Integration

Agents can access knowledge from multiple sources:

### Context Pipeline

```text
┌────────────────────────────────────────────────────────────┐
│                 AGENT CONTEXT PIPELINE                     │
├────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Knowledge   │    │    Web       │    │   Reasoning  │  │
│  │   Lookup     │───▶│   Search     │───▶│    Stage     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │         │
│         ▼                   ▼                    ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  DeepLake +  │    │   arXiv +    │    │   Extended   │  │
│  │   Graphiti   │    │   Semantic   │    │   Thinking   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### Codebase Indexing

Sync your codebase into the knowledge graph:

```yaml
targets:
  - type: process
    name: code_sync
    process:
      type: code_graph_sync
      index_namespace: main
```

This enables agents to:
- Understand model topology
- Query layer dependencies
- Check impact of proposed changes

### Hybrid Storage

```text
┌─────────────────────────────────────────────┐
│                HYBRID STORAGE               │
├─────────────────────────────────────────────┤
│  ┌──────────────┐         ┌──────────────┐  │
│  │   DeepLake   │         │  FalkorDB +  │  │
│  │   (Vector)   │◄───────►│   Graphiti   │  │
│  └──────────────┘         └──────────────┘  │
│        │                        │           │
│        ▼                        ▼           │
│  ┌──────────────┐         ┌──────────────┐  │
│  │    Code      │         │  Entities:   │  │
│  │   Snippets   │         │  - Layers    │  │
│  │  + Summaries │         │  - Functions │  │
│  └──────────────┘         │  - Papers    │  │
│                           └──────────────┘  │
└─────────────────────────────────────────────┘
```

### Setup

```bash
# Start graph services
docker compose up -d falkordb graphiti-mcp

# Run FalkorDB standalone (if not using compose)
docker run -p 6379:6379 -it --rm falkordb/falkordb
```

---

## Code graph ingestion (FalkorDB)

Caramba can maintain a **deterministic structural code graph** (modules/classes/functions/methods + imports/inheritance/basic calls) in FalkorDB.

### Manual sync

```bash
# Uses env FALKORDB_URI if set, otherwise defaults to localhost:6379.
python3.12 -m caramba codegraph-sync .
```

### Git hook auto-sync (recommended)

This repo includes git hooks under `.githooks/`:
- `post-commit`: sync changed `*.py` files after every commit
- `pre-push`: sync changed `*.py` files before every push

Enable them once:

```bash
git config core.hooksPath .githooks
```

Note: if you run FalkorDB via Docker Compose, your container URI is often `redis://falkordb:6379`,
but git hooks run on the host, so they should usually use `redis://localhost:6379`.
You can override the hook target with:

```bash
export CARAMBA_FALKORDB_URI=redis://localhost:6379
```

To temporarily disable syncing:

```bash
export CARAMBA_SKIP_CODEGRAPH_SYNC=1
```

## Configuration

### Team Configuration

Teams map role names to persona configurations:

```yaml
team:
  writer: writer           # Role -> Persona
  reviewer: reviewer
  leader: research_lead
```

Personas are defined in `config/personas/`:

| Persona         | File                | Role              |
|-----------------|---------------------|-------------------|
| `writer`        | `writer.yml`        | Paper drafting    |
| `reviewer`      | `reviewer.yml`      | Paper review      |
| `research_lead` | `research_lead.yml` | Loop coordination |
| `developer`     | `developer.yml`     | Code analysis     |
| `architect`     | `architect.yml`     | System architecture |
| `ml_expert`     | `ml_expert.yml`     | ML insights       |
| `mathematician` | `mathematician.yml` | Formal proofs     |

### Persona Configuration

```yaml
# config/personas/writer.yml
name: writer
description: Academic paper writer

model: gpt-4o
temperature: 0.7

system_prompt: |
  You are an expert academic writer specializing in machine learning.
  Write clear, precise, and well-structured content.
  Use proper mathematical notation where appropriate.

tools:
  - read_tex_file
  - write_tex_file
  - update_section
  - add_citation
  - search_arxiv
  - get_experiment_results
```

### Process Configuration

| Process           | Key Options                                            |
|-------------------|--------------------------------------------------------|
| `paper_write`     | `output_dir`, `writer`                                 |
| `paper_review`    | `strictness`, `max_proposed_experiments`, `output_dir` |
| `research_loop`   | `max_iterations`, `auto_run_experiments`, `output_dir` |
| `code_graph_sync` | `index_namespace`                                      |
| `platform_improve`| `ingest_agent`, `index_namespace`, `ingest_repo`, `ingest_models`, `max_files`, `max_chars_per_file`, `leader_key`, `ideator_keys`, `developer_key`, `reviewer_key`, `repo_root`, `base_branch`, `branch_prefix`, `tests`, `max_review_rounds`, `open_pr`, `pr_title_prefix`, `topic` |

---

## Agent Tools

Agents have access to specialized tools:

### Paper Tools

| Tool                      | Description             |
|---------------------------|-------------------------|
| `read_tex_file`           | Read current paper.tex  |
| `write_tex_file`          | Write complete paper    |
| `update_section`          | Update specific section |
| `add_citation`            | Add BibTeX entry        |
| `search_arxiv`            | Search arXiv for papers |
| `search_semantic_scholar` | Search Semantic Scholar |

### Experiment Tools

| Tool                      | Description                |
|---------------------------|----------------------------|
| `get_experiment_manifest` | Read experiment config     |
| `get_experiment_results`  | Get benchmark results      |
| `list_artifacts`          | List generated files       |
| `include_figure`          | Generate LaTeX figure code |

### Knowledge Tools

| Tool                  | Description            |
|-----------------------|------------------------|
| `search_nodes`        | Query entity graph     |
| `search_memory_facts` | Search relationships   |
| `add_memory`          | Add knowledge to graph |

---

## Running Agent Workflows

### From Manifest

```bash
# Run paper writing process
python3 -m caramba manifest.yml --target paper_write

# Run full research loop
python3 -m caramba manifest.yml --target research_loop
```

### Example Manifest

```yaml
version: 2
name: research_automation

targets:
  # Experiment target
  - type: experiment
    name: train
    # ... experiment config ...

  # Paper writing
  - type: process
    name: write_paper
    team:
      writer: writer
    process:
      type: paper_write
      writer: writer
      output_dir: paper

  # Paper review
  - type: process
    name: review_paper
    team:
      reviewer: reviewer
    process:
      type: paper_review
      reviewer: reviewer
      strictness: conference

  # Full research loop
  - type: process
    name: full_loop
    team:
      leader: research_lead
      writer: writer
      reviewer: reviewer
    process:
      type: research_loop
      leader: leader
      writer: writer
      reviewer: reviewer
      max_iterations: 3

entrypoints:
  default: train
  paper: write_paper
  review: review_paper
  loop: full_loop
```

### Installing Dependencies

```bash
# All agent dependencies
pip install -e ".[agents]"

# Or individual packages
pip install deeplake docling transformers  # Knowledge store
pip install crawl4ai                        # Web crawling
```

---

## Summary

| Process           | Input              | Output                   |
|-------------------|--------------------|--------------------------|
| `paper_write`     | Experiment results | LaTeX paper              |
| `paper_review`    | paper.tex          | Review JSON              |
| `research_loop`   | Manifest           | Iterated paper + reviews |
| `code_graph_sync` | Codebase           | Knowledge graph          |
| `platform_improve`| Telemetry / user feedback | Platform improvements / updated models |

Agent workflows enable:
- 📝 Automated paper generation from experiments
- 🔍 AI-powered review and feedback
- 🔄 Autonomous research iteration
- 🧠 Knowledge-grounded reasoning

---

<div align="center">

**[← Benchmarking](benchmarking.md)** · **[Optimization →](optimization.md)**

</div>
