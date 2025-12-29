# Spec and build

## Configuration
- **Artifacts Path**: {@artifacts_path} → `.zenflow/tasks/{task_id}`

---

## Agent Instructions

Ask the user questions when anything is unclear or needs their input. This includes:
- Ambiguous or incomplete requirements
- Technical decisions that affect architecture or user experience
- Trade-offs that require business context

Do not make assumptions on important decisions — get clarification first.

---

## Workflow Steps

### [x] Step: Technical Specification
<!-- chat-id: 121f3c07-9607-4a82-bfbc-d9ef0d3934d7 -->

Assess the task's difficulty, as underestimating it leads to poor outcomes.
- easy: Straightforward implementation, trivial bug fix or feature
- medium: Moderate complexity, some edge cases or caveats to consider
- hard: Complex logic, many caveats, architectural considerations, or high-risk changes

Create a technical specification for the task that is appropriate for the complexity level:
- Review the existing codebase architecture and identify reusable components.
- Define the implementation approach based on established patterns in the project.
- Identify all source code files that will be created or modified.
- Define any necessary data model, API, or interface changes.
- Describe verification steps using the project's test and lint commands.

Save the output to `{@artifacts_path}/spec.md` with:
- Technical context (language, dependencies)
- Implementation approach
- Source code structure changes
- Data model / API / interface changes
- Verification approach

If the task is complex enough, create a detailed implementation plan based on `{@artifacts_path}/spec.md`:
- Break down the work into concrete tasks (incrementable, testable milestones)
- Each task should reference relevant contracts and include verification steps
- Replace the Implementation step below with the planned tasks

Rule of thumb for step size: each step should represent a coherent unit of work (e.g., implement a component, add an API endpoint, write tests for a module). Avoid steps that are too granular (single function).

Save to `{@artifacts_path}/plan.md`. If the feature is trivial and doesn't warrant this breakdown, keep the Implementation step below as is.

---

### [ ] Step: Update Graph Topology Schema

- Implement `system.graph` topology compatibility with `layers` (keep `nodes` as backward-compatible input).
- Add support for layer-backed graph nodes (reuse `LayerConfig`).

Verification:
- `pytest -q compiler/lower_graph_test.py`

---

### [ ] Step: Integrate Layer Nodes Into GraphSystem

- Teach `model/graph_system.py` to build node modules from either `layer.build()` or `op` strings.
- Ensure repeat expansion deep-copies embedded layer configs.

Verification:
- `pytest -q model/graph_system_test.py`

---

### [ ] Step: Update Lowering, Validation, and Planning

- Update `compiler/lower.py`, `compiler/validate.py`, and `compiler/plan.py` to use the canonical `layers` graph topology schema.
- Add early validation in `config/target.py` for `system.graph` targets.

Verification:
- `pytest -q compiler/lower_manifest_test.py`

---

### [ ] Step: Update Manifests and Documentation

- Update docs to describe `system.graph` and the new graph topology schema.
- If needed, add graph type normalization (`graph` → `GraphTopology`) in `config/resolve.py`.

Verification:
- Manually inspect `docs/manifests.md` and `docs/topologies.md` examples for consistency.

---

### [ ] Step: Run Full Test Suite and Write Report

- Run `pytest -q` (or the closest available equivalent).
- Write `{@artifacts_path}/report.md`:
  - What was implemented
  - How the solution was tested
  - Biggest issues or challenges
