# Alternative-paradigm CPU AVX-512 audit (T2.30)

Maps R15 / manifest alternative-paradigm workloads to `pkg/backend/device/cpu/*` domains and AVX-512 registration per `docs/cpu-dispatch-matrix.md`.

## Summary

| Paradigm / manifest surface | CPU domain | AVX-512 hot path |
|----------------------------|------------|------------------|
| Active inference (FE, EFE, belief, precision) | `active_inference` | yes (`f32_avx512_amd64.s`) |
| Predictive coding | `predictive_coding` | yes |
| Vector symbolic (bind, bundle, permute, similarity) | `vsa` | yes |
| Hawkes / Markov blanket | `hawkes` | yes |
| Causal inference | `causal` | yes |
| Physics stencils / PDE | `physics` | yes |
| EBM free energy (`block.energy.free_energy`) | `math` + `elementwise` | yes (logsumexp, mul) |
| Hebbian optimizer (`train.optimizer.hebbian`) | `optimizer` | yes (`HebbianStepRowFloat32AVX512Asm`) |
| Activation steering / probes | `interpretability` | yes (`ActivationSteerFloat32AVX512Asm`) |
| Model graft injection (`model.graft` read_write) | `model_editing` | yes (`WeightGraftAddFloat32AVX512Asm`) |
| Model freeze / surgery / LoRA | — | graph/metadata; no dedicated f32 bulk kernel in T2.30 |
| Spiking / evolutionary / NCA | — | not on `device.Backend` yet; no CPU domain |

T2.30 adds **`interpretability`** and **`model_editing`** as dedicated operation domains (32 inventoried domains total). Remaining paradigms stay grouped under existing domains until `Backend` grows separate interfaces.

## Kernel contracts

- **`interpretability`:** `dst = base + coefficient * direction` (dest-last `VFMADD231PS`, `K7` masked tail).
- **`model_editing`:** in-place `weights += injection` (`VADDPS`, `K7` masked tail).

Parity at `parity.Lengths` against scalar reference; amd64 `//go:build amd64` compliance tests on `*avx512*` sources.
