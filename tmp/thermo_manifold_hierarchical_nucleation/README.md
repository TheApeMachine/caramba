# Thermo Manifold

A physics-first learning system (no backpropagation) implemented as a thermodynamic dynamical system.

This iteration keeps the sparse **scatter-gather** / **edge-list** mindset, and adds a missing ingredient
for real-time adaptation: **hierarchical nucleation** ("slow particles" / chunks).

## What changed

### Sparse graph topology
- Grammar is stored as a sparse bond graph (no dense V×V matrices).
- All core operations are scatter/gather friendly.

### Hierarchical nucleation
- `HierarchicalSemanticManifold` introduces a second layer of particles:
  - **Token layer:** token → token sparse bonds
  - **Chunk layer:** trigram chunks (slow particles) + chunk → token sparse bonds

Chunks are created by two fully local signals:
1. **Condensation** from high binding energy along token bonds (stability).
2. **Error-driven nucleation** from prediction surprise (metabolic shock).

## Demos

### 1) Unified semantic→bridge→spectral demo

```bash
python -m thermo_manifold.demos.unified_demo
```

### 2) Rule-shift benchmark (metabolic switch)

A continuous stream flips its syntax at `t=shift_at`.
The benchmark records:
- prediction accuracy over time,
- system energy,
- topological entropy,
- number of chunks.

```bash
python -m thermo_manifold.demos.rule_shift_demo --steps 2000 --shift-at 1000 --context-len 6 --dt 0.02 --out-dir ./artifacts
```

Outputs:
- `artifacts/rule_shift.png`
- `artifacts/rule_shift_metrics.json`
