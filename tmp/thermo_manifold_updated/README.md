# Thermo Manifold (updated)

This package implements a thermodynamic / physics-first learning system without backpropagation.

Key updates vs. the previous prototype:

- **No NxM distance matrix by default**: the physics engine operates on an **edge list** (particle→attractor) so subclasses can provide **local neighbor interactions**.
- **Semantic grammar is fully sparse**: transitions are stored in a **sparse bond graph** (no dense V×V matrices).
- **Refactored grammar loop** into three phases:
  1) `update_topology` (nucleation from observation),
  2) `run_metabolism` (bond maintenance / decay),
  3) `propagate_flow` (excitation transport).
- **BridgeManifold** provides an emergent semantic→spectral transduction layer (no hard-coded word→audio lookup).

## Run the demo

```bash
python -m thermo_manifold.demos.unified_demo
```

