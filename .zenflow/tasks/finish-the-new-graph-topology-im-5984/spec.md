# Technical Specification: Finish The New Graph Topology Implementation

## Difficulty
**Medium → hard**.

Reasoning: the current `system.graph` implementation exists and is tested, but “finishing” it implies **schema + manifest compatibility changes** (introducing `layers` and layer-backed graph nodes), plus **cross-cutting updates** (lowering, validation, plan output, docs). Backward-compat concerns (supporting existing `nodes` payloads) increase risk.

---

## Technical Context

### Language / runtime
- Python
- Pydantic v2 models for config validation
- PyTorch (`torch.nn.Module`) as the primary execution backend

### Relevant existing components
- **Tree topology (single-stream)**: `config/topology.py` (topology nodes contain `layers: list[NodeConfig]`)
- **Layer configs + dynamic build**: `config/layer.py` + `config/Config.build()` in `config/__init__.py`
- **Graph topology (named ports / TensorDict DAG)**:
  - Schema: `config/topology_graph.py` (`GraphTopologyConfig` with `nodes`)
  - Runtime executor: `model/graph_system.py`
  - Compile-time helpers: `compiler/lower.py`, `compiler/validate.py`, `compiler/plan.py`
- **Manifest parsing**: `config/manifest.py` (variable substitution + `normalize_type_names` in `config/resolve.py`)

---

## Problem Statement (as observed)

1. The “graph topology” system uses `topology.nodes` and a bespoke node schema (`op` + `config`), while the rest of the platform (and existing documentation around topology) uses the notion of `topology.layers`.
2. The graph executor supports Torch built-ins and `python:` factories, but **does not integrate with the existing LayerConfig/TopologyConfig build mechanism**.
3. Documentation currently covers only the single-stream topology system; manifests/docs do not describe `system.graph`.

Task requirement interpretation:
- “manifests need to be updated to become compatible with having a topology that has layers” → `system.graph.config.topology` should accept `layers` (and likely treat them as the canonical list).
- “finish the new graph topology” → make graph nodes first-class and compatible with existing “layer” configs, and ensure the compiler/planner/docs reflect the final schema.

---

## Implementation Approach

### 1) Make graph topology look like other topologies (`layers`)
Update `GraphTopologyConfig` to support a `layers` list as the primary field.

Backward compatibility strategy:
- Accept both `nodes` and `layers` on input.
- Pick one canonical output form when dumping (recommended: dump `layers`).

This aligns graph topology manifests with existing patterns in `config/topology.py` and docs.

### 2) Allow graph nodes to be “layers” (reuse `LayerConfig`)
Extend graph node schema so a node can be defined in one of two ways:

**A. Layer-backed node (preferred for internal layers)**
- `layer: LayerConfig` (e.g. `{type: LinearLayer, d_in: 4, d_out: 8, bias: true}`)
- Constraints for layer-backed nodes:
  - Require single input and single output key (since Layer modules are single-tensor in/out).
  - Construction uses `LayerConfig.build()`.

**B. Op-backed node (existing behavior)**
- `op: str` + `config: dict[str, Any]`
- Supports Torch `nn.<Op>` and `python:module:Symbol` factories.
- May allow multi-in / multi-out (as today).

This keeps the graph executor flexible while enabling reuse of the existing layer ecosystem.

### 3) Update lowering / validation / planning to match the schema
- **Lowering** (`compiler/lower.py`): operate on `topo.layers` (while accepting `nodes` via parsing). Repeat expansion must deep-copy embedded layer configs.
- **Planning** (`compiler/plan.py`): print `system.graph` topology as `layers` (and include whether a node is `layer`-backed or `op`-backed).
- **Validation** (`compiler/validate.py`):
  - Keep existing checks (unique ids, output key single-producer, acyclic).
  - Add node-level validation for layer-backed nodes (single in/out, required fields).
- **Early manifest validation** (`config/target.py`): parse `GraphTopologyConfig` for `system.graph` targets to surface errors early (parallel to `ModelConfig` strict parse for language_model systems).

### 4) Manifest compatibility updates
- Add graph topology type normalization (if desired): map `type: graph` → `GraphTopology` in `config/resolve.py`.
- Ensure new examples and docs use the canonical schema (`type: GraphTopology`, `layers: [...]`).

### 5) Documentation updates
- `docs/manifests.md`: add a `system.graph` section with a minimal runnable example.
- `docs/topologies.md`: add a “GraphTopology (named-port DAG)” section clarifying the distinction from single-stream topologies.
- Optionally `docs/layers.md`: mention how to embed `LayerConfig` inside graph nodes.

---

## Source Code / File Changes

### Expected to modify
- `config/topology_graph.py` (schema: `layers` + layer-backed nodes + aliases)
- `model/graph_system.py` (build modules from `layer` or `op`; repeat expansion updates)
- `compiler/lower.py` (graph lowering over `layers`)
- `compiler/validate.py` (validate `layers`; add layer-backed node invariants)
- `compiler/plan.py` (render graph topology as layers)
- `config/target.py` (strict validation for `system.graph` topology payload)
- `config/resolve.py` (optional: add `graph` → `GraphTopology` type alias)

### Expected to modify / add tests
- `model/graph_system_test.py` (add test for layer-backed node; update to `layers`)
- `compiler/lower_graph_test.py` (update to `layers`; add repeat test with layer-backed node)
- `compiler/lower_manifest_test.py` (update graph manifest payloads to `layers`)

### Expected to modify docs
- `docs/manifests.md`
- `docs/topologies.md`

---

## Data Model / API / Interface Changes

### `system.graph.config.topology`
- **Input**: accept either `layers` or legacy `nodes`.
- **Canonical output**: prefer `layers` (recommended) to match the rest of the system.

### Graph node schema
- Add `layer: LayerConfig | None`.
- Keep `op: str` + `config: dict[str, Any]` for free-form ops.
- Enforce “exactly one of (`layer`, `op`) must be present”.
- For `layer` nodes, enforce single `in` and `out` key.

---

## Verification Approach

### Targeted unit tests
- `pytest -q model/graph_system_test.py`
- `pytest -q compiler/lower_graph_test.py`
- `pytest -q compiler/lower_manifest_test.py`

### Full suite (if feasible)
- `pytest -q`

---

## Open Questions (need user input)

1. **Canonical key name**: should graph topology serialize as `layers` (recommended) or keep `nodes` for external users?
2. **Backward compatibility window**: do we need to support both `nodes` and `layers` indefinitely, or can `nodes` be deprecated with a warning (docs-only)?
3. **Layer-backed node constraints**: is it acceptable to require single `in`/`out` for `layer` nodes, with multi-in/out reserved for `op` nodes?
4. **Type normalization**: should `type: graph` remain valid (normalized to `GraphTopology`), or should docs enforce `GraphTopology` only?

