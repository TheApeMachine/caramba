
# rez_emergent

Forward-direction update addressing the critique points from the earlier `Thermodynamic Manifold` codebase:

- **Scalability (physics)**: 1D manifolds can enforce a *local interaction horizon* (K≈sqrt(M) neighbors) instead of computing full N×M distances each step.
- **Bridge integrity**: replaces the hard-coded word→audio lookup with an emergent **BridgeManifold** that learns associations through co-activation.
- **Stability**: adds a **Homeostat** (thermostat) that normalizes decay dynamics around the system’s own energy baseline (no fixed target thresholds).
- **Code structure**: semantic learning is explicitly phased (topology → metabolism → drift → heat → nucleation), rather than a monolithic step.

## Run

```bash
python3 unified_demo.py
python3 unified_demo.py --beefy
python3 unified_demo.py --beefy-prob
```

Artifacts:
- `tmp/rez_emergent/unified_demo.wav`
