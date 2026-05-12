# Metal / compute bridge — status (May 2026)

## Done

- **Centralised error labelling**: `metal_error.h` / `metal_error.c` document return-code families.
- **GPU `.metallib` targets** (`Makefile`): `vsa`, `active_inference`, `hawkes`, `predictive_coding`, **`markov_blanket`**, **`causal`**, plus NN stack libraries.
- **Research Metal**: `metal_active_inference.m`, `metal_hawkes.m`, `metal_predictive_coding.m`, `vsa.m`, **`metal_markov_blanket.m`**, **`metal_causal.m`** load their `.metallib` files and dispatch compute (no host-side numerics in those bridges). Hawkes simulate uses float-only on-device RNG (Metal has no `double` in kernels).
- **Markov blanket**: `markov_blanket.metal` — partition (device), flow internal/active (parallel rows), Gaussian MI (device Cholesky / log-det in float).
- **Causal**: `causal.metal` — parallel axpy/sub/matvec, atomic dot reduction, and single-thread kernels with device workspace for do-calculus, backdoor, IV, CATE, DAG–Markov factorisation (acyclicity + per-node OLS + log-score).
- **Thread-safety**: Serial `dispatch_queue` in ObjC bridges; **`sync.Mutex`** on the Go wrappers.
- **Parity** (darwin/cgo): `causal_parity_test.go` uses **`causal.metallib`**; run `make build` first.

## Callers / docs

- Pair each `New*Ops(metallib)` with **`Close()`**; paths must match Makefile outputs (`markov_blanket.metallib`, `causal.metallib`, `hawkes.metallib`, etc.).
- Numerics are **float32** on Metal; parity vs CPU `float64` paths is best-effort within test tolerances.
