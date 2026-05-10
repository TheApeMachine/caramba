I now have a clear picture of the current node graph editor. Here's my assessment:

## Current State

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  NODE GRAPH EDITOR: Current Capabilities                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✓ WORKING                                                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Visual node placement and dragging                                       │
│  • Operation picker (fetches groups from framework/operation/**)            │
│  • Connection drawing between ports                                         │
│  • Named inputs/outputs per operation (from OperationSpec)                  │
│  • Required vs optional inputs distinction                                  │
│  • Missing input badge indicators                                           │
│  • Color-coded ports by data type (Tensor, Mask, etc.)                     │
│  • Load/save to manifest YAML via backend                                   │
│  • Input/Output graph nodes                                                 │
│                                                                             │
│  ✗ MISSING / NEEDED                                                         │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • Node configuration panel (can't set d_model, num_heads, etc.)           │
│  • Shape inference / validation                                             │
│  • Undo/redo                                                                │
│  • Copy/paste nodes                                                         │
│  • Sub-graph grouping (collapse nodes into a "block")                      │
│  • Templates / presets                                                      │
│  • Auto-layout                                                              │
│  • Zoom/pan                                                                 │
│  • Keyboard shortcuts                                                       │
│  • Live validation feedback                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## What It Needs to Become

The editor should be the primary way to build architectures—not a "nice to have" but **the** tool. Here's the vision:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ARCHITECTURE EDITOR: Target State                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FILE   EDIT   VIEW   INSERT   TEMPLATES   VALIDATE   RUN          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────┬─────────────────────────────────┬──────────────────┐     │
│  │              │                                 │                  │     │
│  │  OPERATIONS  │        CANVAS                   │   INSPECTOR      │     │
│  │              │                                 │                  │     │
│  │  ▼ Attention │    ┌─────┐      ┌─────┐       │  ┌────────────┐   │     │
│  │    • SDPA    │    │  x  │─────▶│ W_q │───┐   │  │ W_q        │   │     │
│  │    • DBA     │    └─────┘      └─────┘   │   │  │            │   │     │
│  │    • GQA     │        │        ┌─────┐   │   │  │ d_in: 2048 │   │     │
│  │              │        ├───────▶│ W_k │───┤   │  │ d_out: 2048│   │     │
│  │  ▼ Projection│        │        └─────┘   │   │  │ bias: true │   │     │
│  │    • Linear  │        │        ┌─────┐   │   │  │            │   │     │
│  │    • FusedQKV│        └───────▶│ W_v │───┤   │  │ [Infer]    │   │     │
│  │              │                 └─────┘   │   │  └────────────┘   │     │
│  │  ▼ Normalize │                           ▼   │                    │     │
│  │    • RMSNorm │                      ┌────────┐│  SHAPES           │     │
│  │    • LayerNorm                      │  SDPA  ││  ─────────────    │     │
│  │              │                      └────────┘│  in:  [B,T,2048]  │     │
│  │  ▼ Templates │                           │   │  out: [B,T,2048]  │     │
│  │    • TransformerBlock                    ▼   │                    │     │
│  │    • MLP     │                      ┌─────┐  │  FLOPS: 12.4M     │     │
│  │    • DBA Block                      │ out │  │  Params: 4.2M     │     │
│  │              │                      └─────┘  │                    │     │
│  │              │                                 │                  │     │
│  └──────────────┴─────────────────────────────────┴──────────────────┘     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  VALIDATION                                                          │   │
│  │  ✓ Graph is connected                                                │   │
│  │  ✓ All required inputs provided                                      │   │
│  │  ✓ Shapes are compatible                                             │   │
│  │  ⚠ W_q.d_out (2048) ≠ n_heads × d_head (32 × 64 = 2048) — OK        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features to Add

### 1. Node Configuration Panel

When you click a node, show its configurable parameters:

```
┌────────────────────────────────────────┐
│  LINEAR PROJECTION                     │
│  projection.linear                     │
├────────────────────────────────────────┤
│                                        │
│  d_in         [2048      ] ← inferred  │
│  d_out        [2048      ]             │
│  bias         [✓]                      │
│                                        │
│  ─────────────────────────────────────│
│  Input Shape:  [B, T, 2048]            │
│  Output Shape: [B, T, 2048]            │
│  Parameters:   4,194,304               │
│  FLOPs:        4,194,304 per token     │
│                                        │
│  [Apply] [Reset]                       │
└────────────────────────────────────────┘
```

**Backend requirement:** Extend `/api/operations/spec` to include:
- Parameter definitions (name, type, default, constraints)
- Shape inference rules (output shape as function of inputs + config)
- Parameter count formula

### 2. Shape Inference

As you connect nodes, infer and display shapes:

```python
# Backend: shape inference engine
def infer_shapes(graph: GraphTopology, input_shapes: dict[str, Shape]) -> dict[str, Shape]:
    """
    Given input shapes, propagate through the graph.
    Returns shape at every node output.
    """
    shapes = {}
    
    for node in topological_sort(graph.nodes):
        input_shapes_for_node = [shapes[inp] for inp in node.inputs]
        output_shapes = infer_node_output_shapes(
            op=node.op,
            config=node.config,
            input_shapes=input_shapes_for_node,
        )
        for name, shape in zip(node.outputs, output_shapes):
            shapes[name] = shape
    
    return shapes
```

Frontend displays shapes on hover or in inspector:

```
┌─────┐                    ┌─────┐
│ x   │──[B,T,2048]───────▶│ W_q │──[B,T,2048]──▶
└─────┘                    └─────┘
```

### 3. Templates / Presets

Drag-and-drop common patterns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TEMPLATES                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │             │  │             │  │             │  │             │        │
│  │  Standard   │  │    DBA      │  │    GQA      │  │  MLP Block  │        │
│  │  Attention  │  │  Attention  │  │  Attention  │  │             │        │
│  │             │  │             │  │             │  │             │        │
│  │   ○ ○ ○     │  │   ○─○─○     │  │   ○ ○ ○     │  │   ○───○     │        │
│  │   │╲│╱│     │  │   │ │ │     │  │   │╲│╱│     │  │   │   │     │        │
│  │   ○─○─○     │  │   ○─○─○     │  │   ○─┴─○     │  │   ○───○     │        │
│  │             │  │             │  │             │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
│  Click to insert at cursor, or drag to canvas                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

A template is just a pre-wired sub-graph with configurable variables.

### 4. Sub-Graph Grouping (Blocks)

Select multiple nodes → "Group into Block":

```
BEFORE:                              AFTER:
                                     
  ┌───┐   ┌───┐                        ┌─────────────────┐
  │W_q│   │W_k│                        │                 │
  └─┬─┘   └─┬─┘                        │  "Attention"    │
    │       │      ┌───┐               │                 │
    │       │      │W_v│       ═══▶    │  (12 nodes)     │
    │       │      └─┬─┘               │                 │
    └───┬───┴───────┬┘                 │  [expand]       │
        │           │                  │                 │
      ┌─┴───────────┴─┐                └─────────────────┘
      │     SDPA      │                
      └───────────────┘                
```

Blocks can be:
- Collapsed for cleaner view
- Saved as templates
- Repeated (with weight sharing options)

### 5. Live Validation

As you edit, continuously validate:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  VALIDATION                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✓ Graph connectivity: All nodes reachable from inputs                      │
│  ✓ Required inputs: All operations have required inputs connected           │
│  ✗ Shape mismatch: SDPA expects Q,K,V with same d_head                     │
│     • W_q outputs [B,T,2048] but W_k outputs [B,T,1024]                    │
│     • Did you mean to set W_k.d_out = 2048?                                │
│  ⚠ Unused output: W_v.out is not connected to anything                     │
│                                                                             │
│  [Fix shape mismatch] [Ignore warning]                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6. "Test Run" Mode

Without training, just validate the architecture compiles and runs:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TEST RUN                                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input shape: [1, 128, 2048]  (batch=1, seq=128, d_model=2048)             │
│                                                                             │
│  [Run Test]                                                                 │
│                                                                             │
│  Result:                                                                    │
│  ✓ Compilation successful                                                   │
│  ✓ Forward pass completed                                                   │
│  ✓ Output shape: [1, 128, 2048]                                            │
│                                                                             │
│  Timing:                                                                    │
│  • Compile: 0.23s                                                           │
│  • Forward: 0.008s                                                          │
│                                                                             │
│  Memory:                                                                    │
│  • Parameters: 124.5M (498 MB)                                              │
│  • Activations: 67 MB (batch=1)                                             │
│  • Peak: 612 MB                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implementation Priority

| Feature | Impact | Effort | Priority |
|---------|--------|--------|----------|
| Node configuration panel | Critical | Medium | **1** |
| Shape inference | High | Medium | **2** |
| Undo/redo | High | Low | **3** |
| Live validation | High | Medium | **4** |
| Templates | Medium | Medium | 5 |
| Sub-graph grouping | Medium | High | 6 |
| Test run | Medium | Medium | 7 |
| Copy/paste | Medium | Low | 8 |
| Auto-layout | Low | Medium | 9 |

The **node configuration panel** is the critical missing piece. Without it, you can't actually set parameters, which makes the editor useless for real work.
