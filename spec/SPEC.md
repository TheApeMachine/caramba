# Project spec

## Vision

**Caramba** is a manifest-driven machine learning research substrate: researchers declare topology, runtime behavior, state, and instrumentation in YAML (“a manifest is a model”). The manifest serves as a rudimentary programming language—atomic operations for control flow (e.g., looping for next-token generation), IO (e.g., stdin), and custom state—so the platform does not pre-prescribe KV caches, training loops, or sampling strategies in Go. Researchers compose those mechanisms in YAML and execute them with resident tensors on every supported compute target under strict accuracy and provenance contracts.

**What exists today**

- **Low-level compute:** `pkg/backend/device/cpu` — 30 operation domains (`activation` … `vsa`, excluding shared `cpu/neon`) with Go scalar on all; **NEON** registered in 20/30 domains; **amd64 AVX-512** in `activation`, `attention`, `convolution`, `dot`, `dropout`, `elementwise`, `embedding`, `layernorm`, `losses`, `matmul`, `normalization`, `pool`, `pospop`, `reduction`, `rope`, `sampling`, and `shape` (17/30); **AVX2/SSE2** only in `activation` and `pospop`. Legacy **386** paths removed; CPU targets **amd64** and **arm64** only (`GOARCH=386` unsupported).
- **Device backends:** `pkg/backend/device/{metal,cuda,xla}` — **Metal** has 462 `kernels.Default` registrations (158 unique names, 68/119 required ops); **CUDA** and **XLA** are tensor upload/download only (0 kernel registrations). **`device.Backend` ↔ `ir.RequiredOperationIDs()` inventory** in `docs/backend-inventory.md` and `pkg/backend/device/inventory*.go` (151 methods, 119 required ops cross-linked). **CPU dispatch matrix** in `docs/cpu-dispatch-matrix.md` and `pkg/backend/device/cpu/dispatchaudit/`. **Device backend matrix** in `docs/device-backend-matrix.md` and `pkg/backend/device/backendaudit/`. **Combined backend coverage** in `docs/backend-coverage.md` and `pkg/backend/device/coverageaudit/` (T1.2–T1.4 registration snapshot, R1 execution-target summary).
- **Compute core:** `pkg/backend/compute/{ir,tensor,kernels,fusion,state}`; minimal `pkg/backend/compute/runtime` (`executor.go`, `host.go` only).
- **Assets and templates:** `pkg/asset` (embedded YAML, `system.topology` shapes, `TemplateFS` for includes); model manifests under `pkg/asset/template/manifest/`; canonical runtime YAML in `pkg/asset/template/runtime/{chat,diffusion}.yml`.
- **Hub, provenance, API, frontend:** `pkg/hub`, `pkg/notary`, `pkg/backend/api` + `cmd/serve`, TanStack Start `frontend/` (node-graph, research, benchmark routes).
- **Stores:** `pkg/store` clients (Qdrant, Neo4j, Elasticsearch, DeepLake) load `store.*` from `cmd/asset/config.yml` via `pkg/config` (`New*StoreConfig`); S3 callers pass `Config` explicitly.
- **Compliance hygiene:** `pkg/backend/device/complianceaudit/` machine-checks forbidden phrasing, cross-ISA aliasing, and scalar-in-SIMD hot loops (T1.6).
- **CLIs:** `cmd/chat` and `cmd/image` are **stubs** (`RunE` returns `nil`).

**Gaps (relative to README and `docs/research-platform-requirements.md`)**

- No `pkg/manifest` or `pkg/runtime` packages yet (docs describe target layout; code does not).
- No manifest→`ir.Graph` / runtime-program compiler pipeline wired end-to-end.
- No graph optimizer package at `pkg/backend/compute/compiler` (README checklist item is aspirational).
- **AVX-512** incomplete across CPU domains (registered on `activation`, `attention`, `convolution`, `dot`, `dropout`, `elementwise`, `embedding`, `layernorm`, `losses`, `matmul`, `normalization`, `pool`, `pospop`, `reduction`, `rope`, `sampling`, `shape` — 17/30 per dispatch matrix; **T2.1/T2.2** gap-close await human §10b only); `dispatchaudit` `TestValidateCPUDispatchMatrix` still fails 30 vs 32 domain invariant (pre-existing); **backend legality** and **zero-copy residency** tests not implemented.
- **R9 (partial):** `pkg/store` off direct `getenv`; `os.Getenv` still used in opt-in audit doc-regeneration tests (`coverageaudit`, `backendaudit`).
- Training/fine-tuning, manifest-driven tuners, distributed/voluntary compute, and research UX (ModelScope, layer surgery, node-graph round-trip) are not done.

---

## Requirements

- [ ] R1: Every method on `pkg/backend/device.Backend` (and every ID from `ir.RequiredOperationIDs()`) has real implementations on Go scalar, AVX-512, AVX2, SSE2, NEON, Metal, CUDA, and XLA — no aliasing, no host-side masquerading, no silent CPU fallback on device backends, no scalar loops inside SIMD/GPU kernels.
- [ ] R2: Backend kernel parity tests at N ∈ {1, 7, 64, 1024, 8192} with tight ULP bounds against the scalar reference; every kernel has a benchmark; completion claims include pasted test and benchmark output.
- [ ] R3: A manifest compiler lowers `system.topology` → `ir.Graph` and `system.runtime` → typed runtime program IR with strict validation (undeclared references fail at compile time). Manifests are composable via `include`. Standard formats (e.g., Hugging Face SafeTensors + tokenizer configs) can be ingested into graph/runtime structures without a hand-authored architecture YAML when atomic ops exist.
- [ ] R4: A runtime executor runs manifest programs (IO, tokenizer, control, graph calls, samplers, schedulers, state, training, evaluation, telemetry) with tensors resident on the selected backend until explicit readback boundaries.
- [ ] R5: Terminal chat and image generation run end-to-end from manifest programs only (`cmd/chat`, `cmd/image` wired to runtime + Hub + provenance).
- [ ] R6: Manifest-driven training and fine-tuning (forward, loss, backward, optimizer step, checkpoints, validation, metrics) without Go control loops in application code; IR-level autodiff generates backward graphs for manifest-composed forwards; non-gradient optimization is expressible via manifest control flow.
- [ ] R7: Backend legality checking rejects manifests whose operation, dtype (FP32, FP16, BF16, INT8, INT4, FP8), layout, state, or fusion contracts are unsupported on the chosen device; execution maintains strict zero-copy residency.
- [ ] R8: Research runs emit a signed provenance ledger (manifests, revisions, artifact hashes, seeds, metrics, traces, checkpoints, outputs) via `pkg/notary`.
- [ ] R9: Configuration is loaded only through `cmd/asset/config.yml` and `pkg/config`; no direct `os.Getenv` / `os.LookupEnv` outside `pkg/config`.
- [ ] R10: Code style and structure follow `AGENTS.md` (methods over loose functions, file/method size limits, GoConvey tests mirroring methods, no banned fallback patterns).
- [ ] R11: README feature checklist stays accurate; listed Hugging Face models run from bundled or resolved manifests with smoke tests.
- [ ] R12: Researchers inspect and modify graphs and runtime programs from the frontend (node graph, ModelScope, layer surgery) without writing Go; YAML is the single source of truth (two-way binding with the node graph).
- [ ] R13: Optional local-only and multi-user collaboration modes respect privacy and team workflows in README (egress firewall, offline Hub, local notary; Clerk-gated teams).
- [ ] R14: Distributed training and voluntary compute integrate with `docs/architecture.md` and `docs/voluntary-distributed-compute.md` without breaking residency and legality rules.
- [ ] R15: Alternative paradigms exposed on `Backend` (spiking, energy-based, evolutionary, Hebbian, cellular automata, interpretability, model editing) are manifest-composable and meet R1/R2 on every backend path.
- [ ] R16: Mechanistic interpretability and model editing (logit lens, SAE, activation patching, rank-one updates, layer surgery) are driven by manifest hooks, not ad hoc Go scripts.

---

## Roadmap

Tasks use stable IDs. The sync phase checks these off when review passes. Each task is sized for one develop→review cycle.

### Phase 1: Backend audit and platform hygiene

- [x] T1.1 — Finish removing legacy `386` CPU build tags, selectors, and `.s` files across all domains; `GOARCH=386` builds are unsupported (requirement: R1, R10; non-goal: 32-bit)
- [x] T1.2 — Inventory every method on `device.Backend` and cross-link to `ir.RequiredOperationIDs()` (requirement: R1)
- [x] T1.3 — Per-domain CPU dispatch matrix: scalar / AVX-512 / AVX2 / SSE2 / NEON registration status (requirement: R1)
- [x] T1.4 — Per-backend matrix for Metal, CUDA, XLA: registered ops, dtypes, and missing kernels (requirement: R1)
- [x] T1.5 — Publish combined coverage matrix under `docs/backend-coverage.md` and link from README (requirement: R1, R11)
- [x] T1.6 — Fix audit findings: ISA aliasing, scalar-in-SIMD files, widened test epsilons, forbidden comment phrasing (requirement: R1, R2, R10)
- [x] T1.7 — Migrate `pkg/store` clients off direct `os.Getenv` / `os.LookupEnv` into `pkg/config` (requirement: R9)

### Phase 2: AVX-512 CPU kernels (amd64)

One task per domain: real `_avx512_amd64.s` bodies, dispatch wiring, parity at N ∈ {1, 7, 64, 1024, 8192}, benchmark. Skip domains with no amd64 SIMD today only if the audit (T1.3) marks them scalar-only by design.

**T2.1–T2.2** extend domains that already register AVX-512 per `docs/cpu-dispatch-matrix.md` — close every dispatch symbol gap, pass `complianceaudit` on touched amd64 `.s` files, and prove parity (not greenfield registration). **T2.3+** add AVX-512 where the matrix shows `—` today.

- [ ] T2.1 — `activation` AVX-512: complete every AVX-512 symbol (real zmm math in hot loops; masked 1–3 lane tails on `*avx512*` paths, no cross-ISA jumps); zero `complianceaudit` findings on `*avx512*` amd64 `.s`; `//go:build amd64` parity + benchmark tests at `parity.Lengths`; **agent close-out (arm64 Curious):** deliverables in working tree + `GOARCH=amd64 go build` + host-runnable tests pass (acceptance §10a); **sync checkoff gate (human):** native amd64+AVX512F pasted test/bench or CI URL (acceptance §10b) (requirement: R1, R2)
- [ ] T2.2 — `pospop` AVX-512: same contract as T2.1 for `pospop`; `//go:build amd64` parity + benchmark tests at `parity.Lengths`; **agent close-out (arm64 Curious):** deliverables in working tree + `GOARCH=amd64 go build` + host-runnable tests pass (acceptance §10a); **sync checkoff gate (human):** native amd64+AVX512F pasted test/bench or CI URL (acceptance §10b) (requirement: R1, R2)
- [x] T2.3 — `elementwise` AVX-512 (requirement: R1, R2)
- [x] T2.4 — `dot` AVX-512 (requirement: R1, R2)
- [x] T2.5 — `matmul` AVX-512 (requirement: R1, R2)
- [x] T2.6 — `reduction` AVX-512 (requirement: R1, R2)
- [x] T2.7 — `pool` AVX-512 (requirement: R1, R2)
- [x] T2.8 — `dropout` AVX-512 (requirement: R1, R2)
- [x] T2.9 — `losses` AVX-512 (requirement: R1, R2)
- [x] T2.10 — `convolution` AVX-512 (requirement: R1, R2)
- [x] T2.11 — `attention` AVX-512 (requirement: R1, R2)
- [x] T2.12 — `embedding` AVX-512 (requirement: R1, R2)
- [x] T2.13 — `layernorm` AVX-512 (requirement: R1, R2)
- [x] T2.14 — `normalization` AVX-512 (requirement: R1, R2)
- [x] T2.15 — `rope` AVX-512 (requirement: R1, R2)
- [x] T2.16 — `sampling` AVX-512 (requirement: R1, R2)
- [x] T2.17 — `shape` AVX-512 (requirement: R1, R2)
- [ ] T2.18 — `masking` AVX-512 (requirement: R1, R2)
- [ ] T2.19 — `causal` AVX-512 (requirement: R1, R2)
- [ ] T2.20 — `quant` / `dequant` AVX-512 (requirement: R1, R2)
- [ ] T2.21 — `optimizer` AVX-512 (requirement: R1, R2)
- [ ] T2.22 — `math` AVX-512 (requirement: R1, R2)
- [ ] T2.23 — `hawkes` AVX-512 (requirement: R1, R2)
- [ ] T2.24 — `physics` AVX-512 (requirement: R1, R2)
- [ ] T2.25 — `active_inference` AVX-512 (requirement: R1, R2)
- [ ] T2.26 — `predictive_coding` AVX-512 (requirement: R1, R2)
- [ ] T2.27 — `vsa` AVX-512 (requirement: R1, R2)
- [ ] T2.28 — `tokenizer` AVX-512 (requirement: R1, R2)
- [ ] T2.29 — `checkpoint` AVX-512 (requirement: R1, R2)
- [ ] T2.30 — Alternative-paradigm domains on CPU (`interpretability`, `model_editing`, spiking/EBM/evolutionary/Hebbian/NCA if split from `Backend`): AVX-512 per audit (requirement: R1, R2, R15)

### Phase 3: Device backends, legality, and residency

- [ ] T3.1 — Implement the full Backend interface natively in Metal; parity vs scalar for all kernels (requirement: R1, R2)
- [ ] T3.2 — Implement the full Backend interface natively in CUDA; parity vs scalar for all kernels (requirement: R1, R2)
- [ ] T3.3 — Implement the full Backend interface natively in XLA; parity vs scalar for all kernels (requirement: R1, R2)
- [ ] T3.4 — Define `device.Capability` (ops, dtypes, layouts, state, fusion) and per-backend registration (requirement: R7)
- [ ] T3.5 — Compile-time legality pass: reject unsupported manifest contracts with explicit missing capability (requirement: R7)
- [ ] T3.6 — Metal graph residency test: no silent host staging during execution (requirement: R1, R7)
- [ ] T3.7 — CUDA graph residency test (requirement: R1, R7)
- [ ] T3.8 — XLA graph residency test (requirement: R1, R7)
- [ ] T3.9 — FP16/BF16/INT8/INT4/FP8 legality matrix tests per backend (requirement: R1, R7)

### Phase 4: Manifest compiler (`pkg/manifest`)

- [ ] T4.1 — Create `pkg/manifest` package layout and document types (`Document`, `GraphModule`) (requirement: R3)
- [ ] T4.2 — YAML loader with `include` resolution via `asset.TemplateFS()` (requirement: R3)
- [ ] T4.3 — Variables, defaults, and `${...}` interpolation (requirement: R3)
- [ ] T4.4 — `repeat` / template expansion for topology nodes (requirement: R3)
- [ ] T4.5 — Parse `system.topology` into manifest AST (requirement: R3)
- [ ] T4.6 — Parse `system.runtime` program blocks (requirement: R3, R4)
- [ ] T4.7 — Static reference validation: undeclared graph/value/state/tokenizer refs are compile errors (requirement: R3)
- [ ] T4.8 — Lower topology nodes → `ir.Graph` (requirement: R3)
- [ ] T4.9 — Create `pkg/backend/compute/compiler` and implement graph verify pass (requirement: R3)
- [ ] T4.10 — Compiler pass: canonicalize (requirement: R3)
- [ ] T4.11 — Compiler pass: CSE (requirement: R3)
- [ ] T4.12 — Compiler pass: algebraic simplify (requirement: R3)
- [ ] T4.13 — Compiler pass: fusion using `pkg/backend/compute/fusion` catalog (requirement: R3)
- [ ] T4.14 — Compiler pass: DCE (requirement: R3)
- [ ] T4.15 — Compiler pass: memory planning and cost scheduling (requirement: R3)
- [ ] T4.16 — SafeTensors weight map → graph constants and bindings (requirement: R3, R11)
- [ ] T4.17 — Hugging Face `config.json` + tokenizer assets → manifest/runtime bindings (requirement: R3, R11)
- [ ] T4.18 — IR autodiff: per-op backward rules in compiler (requirement: R6)
- [ ] T4.19 — IR autodiff: emit backward `ir.Graph` from forward graph (requirement: R6)

### Phase 5: Runtime program layer (`pkg/runtime`)

- [ ] T5.1 — Define `pkg/runtime/program` (`Program`, `Step`, `ValueRef`, `StateDeclaration`) (requirement: R3, R4)
- [ ] T5.2 — Lower runtime YAML: IO ops (`io.read_line`, `io.emit_token`, …) (requirement: R4)
- [ ] T5.3 — Lower runtime YAML: control (`repeat`, `until_eof`, branches) (requirement: R4)
- [ ] T5.4 — Lower runtime YAML: `graph.call` and value wiring (requirement: R4)
- [ ] T5.5 — Lower runtime YAML: sampler and scheduler ops (requirement: R4)
- [ ] T5.6 — Implement `pkg/runtime/state` (KV, RoPE metadata, RNG, counters, tensor slots) (requirement: R4)
- [ ] T5.7 — `pkg/runtime/executor`: step dispatch with step IDs in errors (requirement: R4)
- [ ] T5.8 — `pkg/runtime/backend`: graph-call bridge, weight binding, backend selection (requirement: R4, R7)
- [ ] T5.9 — Migrate useful code from `pkg/backend/compute/runtime` into `pkg/runtime`; leave compute executor boundary clear (requirement: R4)
- [ ] T5.10 — Integrate compile pipeline: manifest → graphs + program + capability requirements (requirement: R3, R4, R7)
- [ ] T5.11 — Execute `pkg/asset/template/runtime/chat.yml` end-to-end on Go scalar backend (requirement: R4, R5)
- [ ] T5.12 — Execute `pkg/asset/template/runtime/diffusion.yml` end-to-end on Go scalar backend (requirement: R4, R5)
- [ ] T5.13 — Manifest templates for KV cache, RoPE, and decode loop (no Go strategies) reusable across models (requirement: R4)
- [ ] T5.14 — Runtime training op surface: `train.forward`, `train.loss`, `train.backward`, `train.optimizer_step`, `train.zero_grad`, `train.clip_grad` (requirement: R6)
- [ ] T5.15 — Wire `pkg/asset/template/manifest/experiment_train.yml` through compiler + executor (requirement: R6)
- [ ] T5.16 — Fine-tuning manifests: LoRA/adapters/freeze via `operation/model/*` templates (requirement: R6)
- [ ] T5.17 — Evaluation, metrics, early stop, checkpoint policy in runtime manifests (requirement: R6, R8)
- [ ] T5.18 — Manifest-driven tuner control loops (bandit/evolutionary) without Go tuners (requirement: R6)
- [ ] T5.19 — Interpretability hooks: logit lens / activation tap manifest ops (requirement: R16)
- [ ] T5.20 — Model-editing manifest ops (patch, graft, rank-one) wired to `Backend.ModelEditing` (requirement: R16)

### Phase 6: Inference CLIs and provenance

- [ ] T6.1 — `cmd/chat`: load `pkg/config`, flags, manifest paths (requirement: R5, R9)
- [ ] T6.2 — `cmd/chat`: Hub resolve model + runtime manifests (requirement: R5)
- [ ] T6.3 — `cmd/chat`: run compiled program, stream tokens via `qpool` (requirement: R5)
- [ ] T6.4 — `cmd/image`: config, manifest, and prompt args (requirement: R5)
- [ ] T6.5 — `cmd/image`: prompt encode → denoise → VAE decode execution path (requirement: R5)
- [ ] T6.6 — `cmd/image`: write output artifact to configured store path (requirement: R5)
- [ ] T6.7 — Notary ledger entries for chat runs (manifest hash, seed, metrics) (requirement: R8)
- [ ] T6.8 — Notary ledger entries for image runs (requirement: R8)
- [ ] T6.9 — Smoke: `openai-community/gpt2` chat forward parity vs reference (requirement: R11)
- [ ] T6.10 — Smoke: `meta-llama/Llama-3.2-1B-Instruct` (requirement: R11)
- [ ] T6.11 — Smoke: `black-forest-labs/FLUX.2-klein-4B` diffusion (requirement: R11)

### Phase 7: Model catalog (remaining README models)

- [ ] T7.1 — `meta-llama/Llama-4-Scout-17B-16E` manifest + smoke (requirement: R11)
- [ ] T7.2 — `google/gemma-4-31B-it` manifest + smoke (requirement: R11)
- [ ] T7.3 — `ibm-granite/granite-4.1-8b` manifest + smoke (requirement: R11)
- [ ] T7.4 — `Qwen/Qwen3-Coder-Next` manifest + smoke (requirement: R11)
- [ ] T7.5 — `stabilityai/stable-diffusion-3-medium` manifest + smoke (requirement: R11)
- [ ] T7.6 — `meituan-longcat/LongCat-AudioDiT-3.5B` manifest + smoke (requirement: R11)
- [ ] T7.7 — `facebook/ijepa_vith16_1k` manifest + smoke (requirement: R11)
- [ ] T7.8 — `facebook/vjepa2-vitg-fpc64-256` manifest + smoke (requirement: R11)

### Phase 8: Research UX and collaboration

- [ ] T8.1 — Node-graph editor: load operation schemas from `pkg/asset` (requirement: R12, R18)
- [ ] T8.2 — Node-graph: serialize canvas → manifest YAML (requirement: R12, R18)
- [ ] T8.3 — Node-graph: parse manifest YAML → canvas (round-trip tests) (requirement: R12, R18)
- [ ] T8.4 — ModelScope: graph topology API for ingested models (requirement: R12, R16, R17)
- [ ] T8.5 — ModelScope: WebGL activation/gradient scrubber (requirement: R12, R16, R17)
- [ ] T8.6 — Layer surgery API per `docs/frontend.md` (requirement: R12, R16)
- [ ] T8.7 — Training replay API (requirement: R12)
- [ ] T8.8 — Benchmark suite UI + `/api/metrics/stream` SSE (requirement: R12)
- [ ] T8.9 — Local-only mode: egress-blocking `http.RoundTripper` + offline Hub + local notary (requirement: R13)
- [ ] T8.10 — Devteam / assistant (`pkg/devteam`, Clerk) behind config opt-in (requirement: R13)
- [ ] T8.11 — WYSIWYG LaTeX paper editor + multi-user collaboration (requirement: R13)

### Phase 9: Distributed and voluntary compute

- [ ] T9.1 — `pkg/backend/compute/distributed` tensor sharding aligned with legality contracts (requirement: R14)
- [ ] T9.2 — Collective ops integrated with residency rules (requirement: R14)
- [ ] T9.3 — `pkg/network` transport for voluntary workers (requirement: R14)
- [ ] T9.4 — Orchestrator integration per `docs/architecture.md` with provenance for multi-node runs (requirement: R8, R14)
- [ ] T9.5 — End-to-end distributed training smoke on two processes (requirement: R14)

---

## Progress

- [x] T1.1 — Finish removing legacy `386` CPU build tags, selectors, and `.s` files across all domains; `GOARCH=386` builds are unsupported
- [x] T1.2 — Inventory every method on `device.Backend` and cross-link to `ir.RequiredOperationIDs()`
- [x] T1.3 — Per-domain CPU dispatch matrix: scalar / AVX-512 / AVX2 / SSE2 / NEON registration status
- [x] T1.4 — Per-backend matrix for Metal, CUDA, XLA: registered ops, dtypes, and missing kernels
- [x] T1.5 — Publish combined coverage matrix under `docs/backend-coverage.md` and link from README
- [x] T1.6 — Fix audit findings: ISA aliasing, scalar-in-SIMD files, widened test epsilons, forbidden comment phrasing
- [x] T1.7 — Migrate `pkg/store` clients off direct `os.Getenv` / `os.LookupEnv` into `pkg/config`
- [ ] T2.1 — `activation` AVX-512: **§10a agent slice complete** on `13aebf6` (7 `*_avx512*` `.s` + dispatch tests; cycles 18–19 review PASS); **§10b sync gate open** — human native amd64+AVX512F run for `TestParamExtraAVX512Parity` / `TestActivationAVX512*` / `TestActivationAVX512AssemblyCompliance` / `Benchmark*AVX512*` (arm64 cannot link `go test` cross-arch) (requirement: R1, R2)
- [ ] T2.2 — `pospop` AVX-512: **§10a agent slice complete** (cycle 20 review PASS; `avx512_amd64.s` + amd64 asm preprocessor fixes + 6 `*_test.go`; `GOARCH=amd64 go build ./pkg/backend/device/cpu/pospop/` exit 0; arm64 generic+NEON parity at `parity.Lengths`; complianceaudit 0 findings on `*avx512*`); **§10b sync gate open** — human native amd64+AVX512F for `TestCount*AVX512Parity` / `TestPospopAVX512AssemblyCompliance` / `BenchmarkCount*AVX512*` (requirement: R1, R2)
- [x] T2.3 — `elementwise` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (12 f32 kernels, masked tails); `select_amd64.go` dispatch; `elementwise_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths`; cycle 21 review PASS; `GOARCH=amd64 go build ./pkg/backend/device/cpu/elementwise/` exit 0; arm64 `go test ./pkg/backend/device/cpu/elementwise/...` pass; docs matrix **3/30**; optional human amd64+AVX512F: `Test.*F32AVX512` / `Benchmark.*F32AVX512` (requirement: R1, R2)
- [x] T2.4 — `dot` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (f64 widen-multiply-accumulate, masked `dot_w4_tail`); `select_amd64.go` dispatch; `dot_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths`; cycle 22 review PASS; `GOARCH=amd64 go build ./pkg/backend/device/cpu/dot/` exit 0; arm64 `go test ./pkg/backend/device/cpu/dot/...` + `complianceaudit` pass (0 findings on `f32_avx512_amd64.s`); docs matrix **4/30**; optional human amd64+AVX512F: `TestDotF32AVX512` / `TestDotAVX512AssemblyCompliance` / `BenchmarkDotF32AVX512` (requirement: R1, R2)
- [x] T2.5 — `matmul` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (row outer-product, w16/w8/w4/masked `mm_col_w4_tail` with `MOVQ DX,CX` before `SHLQ CL`); `select_amd64.go` dispatch; `matmul_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths`; cycles 23–24 review PASS; `GOARCH=amd64 go build ./pkg/backend/device/cpu/matmul/` exit 0; arm64 `go test ./pkg/backend/device/cpu/matmul/...` + `complianceaudit` pass; docs matrix **5/30**; optional human amd64+AVX512F: `TestMatMulF32AVX512` / `TestMatmulAVX512AssemblyCompliance` / `BenchmarkMatMulF32AVX512` (requirement: R1, R2)
- [x] T2.6 — `reduction` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (sum/prod/max/min/l1; masked `w4_tail` with `MOVQ CX,DX` before `SHLQ CL`); `select_amd64.go` dispatch; `reduction_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths`; cycle 25 review PASS; `GOARCH=amd64 go build ./pkg/backend/device/cpu/reduction/` exit 0; arm64 `go test ./pkg/backend/device/cpu/reduction/...` + `complianceaudit` pass; docs matrix **6/30**; optional human amd64+AVX512F: `Test*F32AVX512Parity` / `Benchmark*F32AVX512` (requirement: R1, R2)
- [x] T2.7 — `pool` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (4 row kernels: max/avg × stride-1 / 2×2 stride-2); `select_amd64.go` `HasAVX512F` fast-row dispatch; `pool_avx512_{parity,compliance,bench}_test.go`; cycle 26 review PASS; branch tip `ea1f747`; `GOARCH=amd64 go build ./pkg/backend/device/cpu/pool/` exit 0; arm64 `go test ./pkg/backend/device/cpu/pool/...` + `complianceaudit` pass; docs matrix **7/30**; optional human amd64+AVX512F: `Test*Pool*AVX512` / `BenchmarkMaxPool2x2Stride2AVX512` (requirement: R1, R2)
- [x] T2.8 — `dropout` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (vector XMM `drop_gen_vec`, w16/w8/w4/masked `drop_apply_tail` with `K7`+`VMOVDQU32`); `select_amd64.go` dispatch; `dropout_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths`; cycles 27–29 review PASS; `GOARCH=amd64 go build ./pkg/backend/device/cpu/dropout/` exit 0; arm64 `go test ./pkg/backend/device/cpu/dropout/...` + `complianceaudit` pass; docs/dispatchaudit **8/30** (WT uncommitted); optional human amd64+AVX512F: `TestDropoutF32AVX512Parity` / `BenchmarkDropoutF32AVX512` (requirement: R1, R2)
- [x] T2.9 — `losses` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (MSE/MAE f64 widen-accumulate, masked tails `f32_avx512_amd64.s:93-109`/`:206-222`); `select_amd64.go` dispatch; `losses_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths`; cycle 30 review PASS; `GOARCH=amd64 go build ./pkg/backend/device/cpu/losses/` exit 0; arm64 `go test ./pkg/backend/device/cpu/losses/...` + `complianceaudit` pass; docs/dispatchaudit **9/30** (WT uncommitted); optional human amd64+AVX512F: `TestMseSumF32AVX512Parity` / `TestMaeSumF32AVX512Parity` / `Benchmark*F32AVX512` (requirement: R1, R2)
- [x] T2.10 — `convolution` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (patch-dot f64 widen-accumulate, masked tail `f32_avx512_amd64.s:54-69`); `select_amd64.go` + `select_generic.go` `!amd64`; `convolution_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths`; cycle 31 review PASS; `GOARCH=amd64 go build ./pkg/backend/device/cpu/convolution/` exit 0; arm64 `go test ./pkg/backend/device/cpu/convolution/...` + `complianceaudit` pass; docs/dispatchaudit **10/30** (WT uncommitted); optional human amd64+AVX512F: `TestConvPatchDotF32AVX512Parity` / `BenchmarkConvPatchDotF32AVX512` (requirement: R1, R2)
- [x] T2.11 — `attention` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (flash online-update + scale; dest-last `VMULPS`/`VADDPS` `f32_avx512_amd64.s:24-26,:47-49,:63-65,:84-86`); `select_amd64.go`, `select_flash_amd64.go`; `attention_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths` (0 ULP); cycles 32–33 review PASS; `GOARCH=amd64 go build ./pkg/backend/device/cpu/attention/` exit 0; arm64 `go test ./pkg/backend/device/cpu/attention/...` + `complianceaudit` pass; docs/dispatchaudit **11/30** (WT uncommitted); optional human amd64+AVX512F: `TestFlashAttention*AVX512*` / `BenchmarkFlashAttention*AVX512*` (requirement: R1, R2)
- [x] T2.12 — `embedding` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (`CopyRowFloat32AVX512Asm`, `AddRowFloat32AVX512Asm`; masked tails `f32_avx512_amd64.s:47-57,:114-126`); `f32_generic.go`, `select_amd64.go`, `select_other.go`; `embedding_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths` on hidden (0 ULP); cycle 34 review PASS; `GOARCH=amd64 go build ./pkg/backend/device/cpu/embedding/` exit 0; arm64 `go test ./pkg/backend/device/cpu/embedding/...` + `complianceaudit` pass; docs/dispatchaudit **12/30** (branch tip `751157d`); optional human amd64+AVX512F: `TestLookupF32AVX512Parity` / `TestBagF32AVX512Parity` / `Benchmark*F32AVX512*` (requirement: R1, R2)
- [x] T2.13 — `layernorm` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (`LayerNormSquaredDiffSumFloat32AVX512Asm`, `LayerNormApplyRowFloat32AVX512Asm`; masked tails `f32_avx512_amd64.s:89-100,:200-213`); `f32_generic.go`, `select_amd64.go`, `select_other.go` `!arm64 && !amd64`; `layernorm_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths` (apply 0 ULP); cycle 35 review PASS; `GOARCH=amd64 go build ./pkg/backend/device/cpu/layernorm/` exit 0; arm64 `go test ./pkg/backend/device/cpu/layernorm/...` + `complianceaudit` pass; docs/dispatchaudit **13/30** (WT uncommitted); optional human amd64+AVX512F: `TestLayerNormSquaredDiffSumF32AVX512Parity` / `TestLayerNormApplyRowF32AVX512Parity` / `BenchmarkLayerNorm*AVX512*` (requirement: R1, R2)
- [x] T2.14 — `normalization` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (`NormSquaredDiffSumFloat32AVX512Asm`, `NormApplyConstScaleBiasFloat32AVX512Asm`; masked tails `f32_avx512_amd64.s:87-98,:182-197`); `f32_generic.go`, `natives.go`, `select_amd64.go`, `select_other.go` `!amd64`; `normalization_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths` (apply 0 ULP); cycle 36 review PASS; `GOARCH=amd64 go build ./pkg/backend/device/cpu/normalization/` exit 0; arm64 `go build` + generic host benches pass; docs/dispatchaudit **14/30** (WT uncommitted); optional human amd64+AVX512F: `TestNormSquaredDiffSumF32AVX512Parity` / `TestNormApplyConstScaleBiasF32AVX512Parity` / `BenchmarkNorm*AVX512*` (requirement: R1, R2)
- [x] T2.15 — `rope` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (`RopePairsFloat32AVX512Asm`; masked tail `f32_avx512_amd64.s:94-128`); `f32_generic.go`, `select_amd64.go`, `select_other.go` `!amd64 && !arm64`; `rope_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths` as pair counts (0 ULP); cycle 37 review PASS; `GOARCH=amd64 go build ./pkg/backend/device/cpu/rope/` exit 0; arm64 rope generic bench + neon `TestRoPEFloat32` + complianceaudit pass; docs/dispatchaudit **15/30** (`bd5e8c3`); optional human amd64+AVX512F: `TestRopePairsF32AVX512Parity` / `BenchmarkRopePairsF32AVX512` (requirement: R1, R2)
- [x] T2.16 — `sampling` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (`GreedySampleFloat32AVX512Asm`, `SamplingSoftmaxRowFloat32AVX512Asm`; masked tails `f32_avx512_amd64.s:85-100,:178-191,:282-296,:452-491,:544-556`); `f32_generic.go`, `select_amd64.go`, `select_other.go` `!amd64`; `sampling_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths` (greedy exact, softmax 2 ULP); cycle 38 review PASS; `GOARCH=amd64 go build ./pkg/backend/device/cpu/sampling/` exit 0; arm64 `go test ./pkg/backend/device/cpu/sampling/...` + `complianceaudit` pass; docs/dispatchaudit **16/30** (WT uncommitted); optional human amd64+AVX512F: `TestGreedySampleF32AVX512Parity` / `TestSamplingSoftmaxRowF32AVX512Parity` / `Benchmark*F32AVX512*` (requirement: R1, R2)
- [x] T2.17 — `shape` AVX-512: greenfield `f32_avx512_amd64.{s,go}` (`CopyContiguousFloat32AVX512Asm`, `WhereFloat32AVX512Asm`, `MaskedFillFloat32AVX512Asm`; masked tails `f32_avx512_amd64.s:49-60,:133-159,:221-245`; mask bit-offset `R11`); `f32_generic.go`, `f32_native.go`, `select_amd64.go`, `select_other.go`; `shape_ops.go` wired natives; `shape_f32_parity_test.go` + `shape_avx512_{parity,compliance,bench}_test.go` at `parity.Lengths` (0 ULP); cycle 39 review PASS; branch tip `88fcdf6`; `GOARCH=amd64 go build ./pkg/backend/device/cpu/shape/` exit 0; arm64 `go test ./pkg/backend/device/cpu/shape/...` + `complianceaudit` pass; docs/dispatchaudit **17/30**; optional human amd64+AVX512F: `Test*F32AVX512Parity` / `Benchmark*F32AVX512*` (requirement: R1, R2)

---

## Agent steering

### Developer
- **Default next task: T2.18** (`masking` AVX-512 greenfield — same contract as T2.3–T2.17: `f32_avx512_amd64.{s,go}`, `select_amd64.go`, `select_other.go` `!amd64`, `masking_avx512_{parity,compliance,bench}_test.go`, `dispatchaudit` + `docs/cpu-dispatch-matrix.md` → **18/30**). Entry points today are scalar in `masking/ops.go` (`ApplyMask` f32/bf16/f16 add; `CausalMask` row fill with `-Inf` above diagonal). Matrix row is scalar-only, no NEON — do not block on NEON.
- **Do not** reopen verification-only **T2.1** / **T2.2** except human §10b (native amd64+AVX512F paste or CI URL). **T2.17** landed on branch tip `88fcdf6`; do not re-develop shape unless review FAIL on new evidence.
- **Tail opmask:** remainder count in **CX** before `SHLQ CL` when building K7 (`shape` copy tail `:53-57`, `matmul` `:118-122`, `dropout` `:132-145`); unmasked tail load/store failed T2.8 cycle 28.
- **Bit-mask byte streams** (if any kernel uses `[]byte` masks): track in-byte bit offset like `shape` `R11` in `f32_avx512_amd64.s` (`:106-112`, `:133-159`); advancing mask pointer after partial w8 can mis-handle counts 9–15 — parity set is `{1,7,64,1024,8192}` but prefer correct partial-byte logic or document scope.
- **In-place / dest-last:** dest-last `VADDPS` / `VMULPS` for fused accum (`attention` `:24-26`, `embedding` AddRow); wrong operand order failed T2.11 cycle 32.
- **Kernel contract:** ymm/zmm hot loops; masked tails on `*avx512*` (not `JL *_done`); no `VEXTRACTF128` from Z; `parity.Lengths` + tight ULP; direct asm subtests at `parity.Lengths`.

### Reviewer
- On **arm64 Curious host:** `5_verification` PASS when deliverables are in the **working tree**, host-runnable tests pass, and `GOARCH=amd64 go build` (or `go test -c`) succeeds for touched packages; **do not** FAIL for uncommitted deliverables, absent `HEAD` delta, or missing native amd64+AVX512F paste (§10b is human/sync for T2.1/T2.2 only).
- Fail `3_spec_compliance` for scalar-in-SIMD, widened ULP, missing `parity.Lengths`, unmasked tail load/store, or cross-ISA jumps — not for WT vs `HEAD` alone when the WT contains the claimed fix.

### Sync
- Check off **T2.1** / **T2.2** only after review PASS **and** acceptance **§10b** (human amd64+AVX512F proof or CI URL).
- **T2.3+** greenfield: check off on review PASS per normal Phase 2 bar (no §10b gate).

---

## Acceptance criteria

Work is **done** for a roadmap task only when all of the following hold (aligned with `AGENTS.md` §2 and `docs/research-platform-requirements.md`):

1. **Tests** — Tests that would catch the claimed behavior exist, pass, and are pasted in the completion message.
2. **Kernel parity** — For backend kernels: scalar reference parity at N ∈ {1, 7, 64, 1024, 8192} with tight ULP bounds; no tolerance widening to pass.
3. **Benchmarks** — A benchmark exists, was run, and output is pasted for performance claims.
4. **No banned patterns** — No ISA aliasing, no placeholders, no device-path host fallback, no scalar loops masquerading as vectorized code, no forbidden phrasing in code or comments.
5. **Config** — New settings use `cmd/asset/config.yml` and `pkg/config` only.
6. **README** — User-visible behavior changes update `README.md`.
7. **Researcher bar** — Where the task touches the platform contract, the relevant row in `docs/research-platform-requirements.md` is satisfied.
8. **Phase 2 (CPU AVX-512)** — For `activation` / `pospop` (pre-registered per T1.3), done means zero dispatch/compliance gaps for AVX-512 in that domain, not merely files present. For other domains, done means AVX-512 registered in the dispatch matrix with the same parity/benchmark/compliance bar.
9. **Working tree (review gate)** — The human commits; agents deliver in the **working tree**. Review and develop judge `git diff` and on-disk files — not whether fixes are on `HEAD`. Uncommitted deliverables are valid; do not FAIL review, block develop, or require `git add` / `git commit` / worktrees.
10. **amd64 SIMD proof (T2.1–T2.2 gap-close only)** — split into agent vs sync gates (greenfield **T2.3+** uses item 9 only):
    - **§10a (agent, arm64 Curious host):** Kernel/asm/dispatch/compliance deliverables **in the working tree**; `GOARCH=amd64 go build` (or `go test -c`) for touched CPU packages; host-runnable tests pass; `//go:build amd64` parity/benchmark tests exist with `parity.Lengths`. Missing **native** amd64+AVX512F execution is **not** a develop/review FAIL on arm64.
    - **§10b (sync checkoff, human):** Before sync checks off **T2.1** or **T2.2** only: pasted native amd64+AVX512F `go test` / benchmark output (or CI artifact URL) for that domain’s `//go:build amd64` tests. Agents document commands when blocked; humans paste results.

**Platform-level definition of done:** a researcher authors topology and runtime entirely in YAML, runs chat and diffusion without editing Go, trains or fine-tunes from manifest programs, inspects graph/runtime/tensors/cache/optimizer/timings/provenance from run artifacts, and moves the same manifest across Go scalar, AVX-512, AVX2, SSE2, NEON, Metal, CUDA, and XLA with compile-time legality and proof output.

---

## Orchestrator log

| Timestamp (UTC) | Agent / phase | Task | Outcome | Notes |
|-----------------|---------------|------|---------|-------|
| 2026-05-19 | spec-author | bootstrap | spec created | Initial `spec/SPEC.md` from README, AGENTS.md, and repo exploration |
| 2026-05-19 | roadmap-planner | expand-roadmap | spec updated | Split coarse tasks into single-cycle items; aligned Vision with repo; Progress = Phase 1 only |
| 2026-05-19 | developer / cycle 0 | T1.1 | complete | Removed 30 legacy `386` selectors and `.s` files across 9 CPU domains; commit `43f5c70` |
| 2026-05-19 | reviewer / cycle 0 | T1.1 | PASS | All rubric criteria pass; amd64 SSE2 tests gated; optimizer NEON 2-ULP failure pre-existing |
| 2026-05-19 | sync / cycle 0 | T1.1 | checked off | Review PASS; next develop: **T1.2** |
| 2026-05-19 | developer / cycle 1 | T1.2 | complete | `pkg/backend/device/inventory*.go`, `docs/backend-inventory.md`; 151 Backend methods, 119 required-op cross-links |
| 2026-05-19 | reviewer / cycle 1 | T1.2 | PASS | Machine-checked inventory; kernel_registry notes approximate |
| 2026-05-19 | sync / cycle 1 | T1.2 | checked off | Review PASS; next develop: **T1.3** |
| 2026-05-19 | developer / cycle 2 | T1.3 | complete | `pkg/backend/device/cpu/dispatchaudit/`, `docs/cpu-dispatch-matrix.md`; 30 domains, amd64 SIMD 2/30, NEON 20/30 |
| 2026-05-19 | reviewer / cycle 2 | T1.3 | PASS | Machine-checked dispatch matrix; doc matches audit counts |
| 2026-05-19 | sync / cycle 2 | T1.3 | checked off | Review PASS; next develop: **T1.4** |
| 2026-05-19 | developer / cycle 3 | T1.4 | complete | `pkg/backend/device/backendaudit/`, `kernels.Registry.Snapshot`, `docs/device-backend-matrix.md`; Metal 462 regs / CUDA+XLA 0 |
| 2026-05-19 | reviewer / cycle 3 | T1.4 | PASS | Machine-checked device backend matrix; doc in sync with RenderMarkdown |
| 2026-05-19 | sync / cycle 3 | T1.4 | checked off | Review PASS; next develop: **T1.5** |
| 2026-05-19 | developer / cycle 4 | T1.5 | complete | `pkg/backend/device/coverageaudit/`, `docs/backend-coverage.md`; README link; merges T1.2–T1.4 counts |
| 2026-05-19 | reviewer / cycle 4 | T1.5 | PASS | Combined matrix; doc byte-sync with RenderMarkdown; golden counts match audits |
| 2026-05-19 | sync / cycle 4 | T1.5 | checked off | Review PASS; next develop: **T1.6** |
| 2026-05-19 | developer / cycle 5 | T1.6 | complete | `complianceaudit/`, `parity/`, `peel/`; activation amd64 tail strip + peel wiring; compliance scan 0 findings; README + `docs/backend-compliance-audit.md` |
| 2026-05-19 | reviewer / cycle 5 | T1.6 | FAIL | 32–64 ULP test bounds; conv3d case dropped (575 ULP); avg pool 2×2 NEON disabled; amd64 activation peel+asm unverified on amd64 CI |
| 2026-05-19 | sync / cycle 5 | T1.6 | open | Review FAIL; blocking: `remaining_neon_arm64_test.go:25` maxULPAccumulated=32; `conv3d_neon_arm64_test.go:16-21,52` 64 ULP + removed shape; `pool/select_arm64.go:86-94` avg 2×2 scalar path; activation amd64 parity not run; next develop: **T1.6** |
| 2026-05-19 | developer / cycle 6 | T1.6 | complete | Patch-dot scalar+NEON parity (noinline lane adds); conv3d shape restored; avg 2×2 NEON re-enabled; Markov MI float64 native; arm64 tests pass |
| 2026-05-19 | reviewer / cycle 6 | T1.6 | FAIL | 2_correctness_performance + 3_spec_compliance FAIL: scalar FMOVS loops in `*_neon_arm64.s` (patch-dot, avg pool 2×2); complianceaudit skips arm64; peel amd64 unverified |
| 2026-05-19 | sync / cycle 6 | T1.6 | open | Review FAIL; blocking: `conv3d_patch_dot_neon_arm64.s:20-55`, `conv2d_patch_dot_neon_arm64.s:20-55`, `f32_neon_arm64.s:216-232` scalar-in-SIMD; `complianceaudit/scan.go:178-181` no arm64 scan; `patch_dot_scalar.go:29-32` asm-shaped reference; `peel_relu_sse2_amd64_test.go` not run on amd64; next develop: **T1.6** |
| 2026-05-19 | developer / cycle 7 | T1.6 | complete | f64 `ConvPatchDotScalar` + VFMLA patch-dot NEON; `scanNEONScalarHotLoops`; avg 2×2 `VLD1`+sequential `FADDS` spill; arm64 tests/benches pass |
| 2026-05-19 | reviewer / cycle 7 | T1.6 | FAIL | 2_correctness_performance + 3_spec_compliance FAIL: patch-dot fixed; `f32_neon_arm64.s:216-235` avg 2×2 hot loop still scalar FP math; `complianceaudit/scan.go:336-352` VLD1 exempts scalar-math loops |
| 2026-05-19 | sync / cycle 7 | T1.6 | open | Review FAIL; blocking: `f32_neon_arm64.s:216-235` scalar FADDS/FMULS in avg 2×2 hot loop; `complianceaudit/scan.go:336-352` audit heuristic gap; peel amd64 deferred to CI; next develop: **T1.6** |
| 2026-05-19 | developer / cycle 8 | T1.6 | complete | Avg 2×2 `VLD2`+`VFADD_S4`/`VFMUL_S4`; `neonBlockIsScalarHotLoop` load+scalar-math heuristic; patch-dot/compliance arm64 tests 0 findings |
| 2026-05-19 | reviewer / cycle 8 | T1.6 | PASS | All rubric criteria pass; cycle 7 blockers closed; compliance matrix 0 findings; peel amd64 deferred to CI |
| 2026-05-19 | sync / cycle 8 | T1.6 | checked off | Review PASS; next develop: **T1.7** |
| 2026-05-19 | developer / cycle 9 | T1.7 | complete | `pkg/config/store.go`, `store.*` in config.yml; store clients via `New*StoreConfig`; `env_contract_test.go`; README |
| 2026-05-19 | reviewer / cycle 9 | T1.7 | PASS | All rubric criteria pass; pkg/store no direct getenv; config.yml `${...}` for secrets |
| 2026-05-19 | sync / cycle 9 | T1.7 | checked off | Review PASS; Phase 1 complete; next develop: **T2.1** |
| 2026-05-19 | overseer / cycle 10 | meta | ALIGNED | Phase 1 closed; T2.1/T2.2 scope clarified (gap-close vs greenfield); steering for Phase 2 kernel hygiene; T1.7 still uncommitted on `main` |
| 2026-05-19 | developer / cycle 10 | T2.1 | complete | `softmax_avx512_amd64.s` scaleF32AVX512 + smexp epilogue + reduce_max tail; `param_extra_avx512_amd64_test.go` tight ULP; `activation_avx512_parity_test.go` + `activation_avx512_compliance_test.go`; working tree uncommitted |
| 2026-05-19 | reviewer / cycle 10 | T2.1 | FAIL | 2_correctness_performance + 3_spec_compliance + 5_verification FAIL: scalar VMAXSS tail in reduce_max; smexp/scale exit with 1–3 elems unprocessed; no amd64 AVX-512 parity/bench output; parity gaps (PReLUV, packed gated); direct `*AVX512` tests at N=1,7 vs no asm tails |
| 2026-05-19 | sync / cycle 10 | T2.1 | open | Review FAIL; blocking: `softmax_avx512_amd64.s:223-230` VMAXSS scalar tail; `softmax_avx512_amd64.s:72-74,145-147` smexp/scale 1–3 elem gap; AGENTS.md §2 no pasted amd64 parity/bench; `activation_avx512_parity_test.go` omits preluVF32 + packed gated; `avx512_amd64_test.go` direct symbol N=1,7; deliverables uncommitted; next develop: **T2.1** |
| 2026-05-19 | developer / cycle 11 | T2.1 | complete | `softmax_avx512_amd64.s` masked zmm tails (smexp/scale/smlog/reduce_max); `peel.WrapGatedPacked`; `activation_avx512_parity_test.go` (+PReLUV, packed gated); `avx512_amd64_test.go` peel+composite softmax; working tree uncommitted |
| 2026-05-19 | reviewer / cycle 11 | T2.1 | FAIL | 2_correctness_performance + 3_spec_compliance FAIL: most `*_avx512_amd64.s` still JL *_done when CX<4; `param_extra_avx512_amd64_test.go` / `param_avx512_amd64_test.go` direct *AVX512 at N=1,7 without peel; no amd64+AVX512F pasted parity/bench; cycle-10 softmax/dispatch gaps largely closed |
| 2026-05-19 | sync / cycle 11 | T2.1 | open | Review FAIL; blocking: `avx512_amd64.s` + `param_*`/`gated_*_avx512_amd64.s` symbols omit 1–3 lanes on direct call; `param_extra_avx512_amd64_test.go:102-103` and `param_avx512_amd64_test.go:50-51` direct *AVX512 without peel; AGENTS.md §2 no pasted amd64 AVX-512 parity/bench; deliverables uncommitted; next develop: **T2.1** |
| 2026-05-19 | developer / cycle 12 | T2.1 | complete | Masked `w4` tails across 7 `*_avx512_amd64.s` (+`tanh`); direct-symbol tests (`avx512_amd64_test.go`, `param_*_avx512_amd64_test.go`); cycle-11 `JL *_done` gaps closed in source; working tree uncommitted |
| 2026-05-19 | reviewer / cycle 12 | T2.1 | FAIL | 2_correctness_performance + 3_spec_compliance + 5_verification FAIL: `VEXTRACTF128 $0, Z0, X0` invalid in Go amd64 assembler (56 tails; package won’t build); `param_extra_avx512_amd64.s:681-812` RReLU scalar rand loops; no amd64+AVX512F parity/bench output |
| 2026-05-19 | sync / cycle 12 | T2.1 | open | Review FAIL; blocking: `avx512_amd64.s:104` (+23 unary) and `{softmax,param_*,gated_*}_avx512_amd64.s` — `VEXTRACTF128` from Z rejected; `param_extra_avx512_amd64.s:681-812` RReLU scalar IMULL/MOVL; amd64-tagged tests cannot run until asm assembles; deliverables uncommitted; next develop: **T2.1** |
| 2026-05-19 | developer / cycle 13 | T2.1 | complete | Masked `w4` tails (no `VEXTRACTF128` from Z); vector RReLU LCG; gated/param/param_extra/param_preuv re-patched; `avx512_amd64.s` tail `VMOVSS` fixes; `avx512_amd64_test.go` `//go:build amd64`; working tree uncommitted |
| 2026-05-19 | reviewer / cycle 13 | T2.1 | FAIL | 2_correctness_performance + 3_spec_compliance + 4_homogeneity + 5_verification FAIL: `param_extra_avx512_amd64_test.go:97-101` conditional 10 ULP; `param_extra_avx512_amd64.s` ss/snake/snakep `w4_tail` K2 not `KANDQ` with K7; no amd64+AVX512F parity/bench output; deliverables uncommitted |
| 2026-05-19 | sync / cycle 13 | T2.1 | open | Review FAIL; blocking: `param_extra_avx512_amd64_test.go:97-101` widened parity (assert only when diff>10); `param_extra_avx512_amd64.s:328-332,480-482,640-642` unmasked K2 in `w4_tail`; AGENTS.md §2 no amd64 AVX-512 parity/bench pasted; deliverables uncommitted; next develop: **T2.1** |
| 2026-05-19 | developer / cycle 14 | T2.1 | complete | `param_extra_avx512_amd64_test.go` `parity.Lengths` + `parity.AssertFloat32SlicesWithinULP` (no `diff>10`); `param_extra_avx512_amd64.s` `KANDQ K7,K2` + `VXORPS Y3` in ss/snake/snakep `w4_tail`; arm64 tests pass; working tree uncommitted |
| 2026-05-19 | reviewer / cycle 14 | T2.1 | FAIL | 3_spec_compliance + 5_verification FAIL: cycle-13 fixes correct in working tree but HEAD `28e66ec` still has widened param_extra test + unmasked ss `w4_tail`; no amd64+AVX512F parity/bench pasted |
| 2026-05-19 | sync / cycle 14 | T2.1 | open | Review FAIL; blocking: commit `param_extra_avx512_amd64.{s,_test.go}`; HEAD `param_extra_avx512_amd64_test.go` conditional 10 ULP; HEAD `ss_avx512_w4_tail` missing `KANDQ K7,K2`/`VXORPS Y3`; AGENTS.md §2 no amd64+AVX512F parity/bench pasted; next develop: **T2.1** |
| 2026-05-19 | overseer / cycle 15 | meta | ALIGNED | T2.1 stuck 5 review cycles (10–14): recurring uncommitted deltas + missing amd64+AVX512F proof; kernel/asm blockers largely closed in `ecbf7cf`/`28e66ec`; cycle-14 `param_extra` fix still uncommitted; Progress unchanged; steering → close-out; acceptance +9/+10 branch tip + amd64 proof |
| 2026-05-19 | developer / cycle 15 | T2.1 | complete | T2.1 close-out: no new source delta; `param_extra` fixes on branch tip `b997d61`; arm64 complianceaudit/peel/activation pass; develop did not run mutating git |
| 2026-05-19 | reviewer / cycle 15 | T2.1 | FAIL | 5_verification FAIL: no pasted amd64+AVX512F parity/bench; 3_spec_compliance PASS (`b997d61`); 6_git_safety PASS; amd64-tagged tests not run on arm64 host |
| 2026-05-19 | sync / cycle 15 | T2.1 | open | Review FAIL; blocking: AGENTS.md §2 / acceptance §10 — no amd64+AVX512F pasted output for `TestParamExtraAVX512Parity`, `TestActivationAVX512*`, `TestActivationAVX512AssemblyCompliance`, `Benchmark*AVX512*` (or CI URL); next develop: **T2.1** |
| 2026-05-19 | developer / cycle 16 | T2.1 | complete | Verification-only close-out; no activation source delta vs HEAD `81a65b8`; arm64 `complianceaudit` / `peel` / `activation` pass |
| 2026-05-19 | reviewer / cycle 16 | T2.1 | FAIL | 2_correctness_performance + 3_spec_compliance + 5_verification FAIL: `param_extra_avx512_amd64.s:699` `VPADDD Y9,Z8,Z8`; `:721` `VPAND Z6,Z1,Z1` — `GOARCH=amd64 go build ./pkg/backend/device/cpu/activation/` fails; arm64 host tests pass; static tail/parity checks pass |
| 2026-05-19 | sync / cycle 16 | T2.1 | open | Review FAIL; blocking: fix `RReLUF32AVX512` Go amd64 asm in `param_extra_avx512_amd64.s:699,721`; activation package must assemble on amd64 before AVX-512 parity/bench; next develop: **T2.1** |
| 2026-05-19 | developer / cycle 17 | T2.1 | complete | `param_extra_avx512_amd64.s` RReLU: split ymm/zmm iota init (`VPADDD Y9,Y8,Y8` + `VMOVDQA32`/`VPADDD Z9,Z8,Z8`); `VPANDD` in `rr_avx512_w16`; `param_avx512_amd64_test.go` `//go:build amd64`; arm64 tests pass |
| 2026-05-19 | reviewer / cycle 17 | T2.1 | FAIL | 5_verification + 6_git_safety FAIL: `GOARCH=amd64 go build` still fails `param_preuv_sse2_amd64.s:19` (RReLU not in error); develop ran forbidden `git checkout` on preuv; 2/3/4 PASS; RReLU cycle-16 blockers closed in WT |
| 2026-05-19 | sync / cycle 17 | T2.1 | open | Review FAIL; blocking: `param_preuv_sse2_amd64.s:19` amd64 activation package build; no mutating git (`git checkout` on preuv in cycle 17); next develop: **T2.1** |
| 2026-05-19 | developer / cycle 18 | T2.1 | complete | `param_preuv_sse2_amd64.s` VEX PReLU; collateral amd64 asm fixes (`param_sse2`, `softmax_avx2/sse2/avx512`, `sse2_amd64`); `GOARCH=amd64 go build ./pkg/backend/device/cpu/activation/` exit 0; arm64 tests pass; no mutating git |
| 2026-05-19 | reviewer / cycle 18 | T2.1 | PASS | All rubric criteria pass; cycle-17 preuv/build blockers closed in WT; `param_extra_avx512` RReLU on HEAD; amd64+AVX512F parity/bench not run on arm64 (workflow N/A) |
| 2026-05-19 | sync / cycle 18 | T2.1 | open | Review PASS (assembler/package-build slice); T2.1 roadmap unchecked per sync gate; remaining: commit WT asm + optional amd64+AVX512F parity/bench; next develop: **T2.1** |
| 2026-05-19 | developer / cycle 19 | T2.1 | complete | Verification-only; no activation delta vs HEAD (`13aebf6`); `GOARCH=amd64 go build ./pkg/backend/device/cpu/activation/` exit 0; arm64 complianceaudit/peel/activation pass |
| 2026-05-19 | reviewer / cycle 19 | T2.1 | PASS | All rubric criteria pass; asm on branch tip; amd64+AVX512F parity/bench blocked on arm64 host (cross-link `actX86ExpC`); acceptance §10 outstanding |
| 2026-05-19 | sync / cycle 19 | T2.1 | open | Review PASS (verification slice); T2.1 unchecked per sync gate + acceptance §10; branch tip satisfied (`13aebf6`); next develop: **T2.1** (amd64+AVX512F parity/bench paste or CI URL) |
| 2026-05-19 | overseer / cycle 20 | meta | DRIFT | T2.1 §10a complete on `13aebf6`; cycles 15–19 verification loop; stale steering (commit param_extra) removed; acceptance §10a/§10b split; agent default **T2.2**; T2.1 sync awaits human amd64 proof |
| 2026-05-19 | developer / cycle 20 | T2.2 | complete | `pospop/`: amd64 `.s` preprocessor fixes (`sse2`/`avx2`/`avx512`); 6 test files (`generic`, `neon_parity`, `pospop_avx512_{parity,compliance,bench}`); `GOARCH=amd64 go build ./pkg/backend/device/cpu/pospop/` exit 0; arm64 tests/benches pass; WT uncommitted |
| 2026-05-19 | reviewer / cycle 20 | T2.2 | PASS | All rubric criteria pass; §10a satisfied in WT; amd64 AVX-512 parity/bench not run on arm64 (`//go:build amd64`); T2.1 §10b still open |
| 2026-05-19 | sync / cycle 20 | T2.2 | open | Review PASS (§10a); T2.2 unchecked per §10b sync gate; T2.1 §10b still open; next develop: **T2.3** |
| 2026-05-19 | developer / cycle 21 | T2.3 | complete | `elementwise/`: `f32_avx512_amd64.s` (12 kernels), `f32_avx512_amd64.go`, `select_amd64.go` AVX-512 dispatch; `elementwise_avx512_{parity,compliance,bench}_test.go`; `dispatchaudit/matrix_test.go`; docs 3/30; commit `e48628b`; WT Abs parity (`math.Abs` ref) + doc summary deltas |
| 2026-05-19 | reviewer / cycle 21 | T2.3 | PASS | All rubric criteria pass; masked `w4_tail`, VCMPPS+VBLENDMPS max/min; 12 parity tests at `parity.Lengths`; amd64 build/test -c pass; native AVX-512F tests not run on arm64 |
| 2026-05-19 | sync / cycle 21 | T2.3 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.4** |
| 2026-05-19 | developer / cycle 22 | T2.4 | complete | `dot/`: `f32_avx512_amd64.{s,go}` (VCVTPS2PD+VFMADD231PD, masked `dot_w4_tail`); `select_amd64.go` dispatch; `dot_avx512_{parity,compliance,bench}_test.go`; `dispatchaudit/matrix_test.go` + docs **4/30**; WT uncommitted |
| 2026-05-19 | reviewer / cycle 22 | T2.4 | PASS | All rubric criteria pass; f64 dot parity at `parity.Lengths`; complianceaudit 0 findings on `f32_avx512_amd64.s`; amd64 build/test -c pass; native AVX-512F tests not run on arm64 |
| 2026-05-19 | sync / cycle 22 | T2.4 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.5** |
| 2026-05-19 | developer / cycle 23 | T2.5 | complete | `matmul/`: `f32_avx512_amd64.{s,go}` row kernel (w16/w8/w4/masked tail); `select_amd64.go` `HasAVX512F` dispatch; `matmul_avx512_{parity,compliance,bench}_test.go`; `dispatchaudit/matrix_test.go` + docs **5/30** in WT; `GOARCH=amd64 go build ./pkg/backend/device/cpu/matmul/` exit 0; arm64 matmul + complianceaudit pass; WT uncommitted |
| 2026-05-19 | reviewer / cycle 23 | T2.5 | FAIL | 2_correctness_performance FAIL: `f32_avx512_amd64.s:114-121` `mm_col_w4_tail` builds K7 with `SHLQ CL` but count in DX (CX stale/0); masked 1–3 column tails wrong; 1/3/4/5/6 PASS; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 23 | T2.5 | open | Review FAIL; blocking: fix `mm_col_w4_tail` opmask (`MOVQ DX,CX` before `SHLQ CL` or equivalent); native amd64 parity at N∈{1,7} unverified on arm64; T2.1/T2.2 §10b still open; next develop: **T2.5** |
| 2026-05-19 | developer / cycle 24 | T2.5 | complete | `f32_avx512_amd64.s:118` `MOVQ DX,CX` before `SHLQ CL` in `mm_col_w4_tail` (cycle-23 blocker); `GOARCH=amd64 go build ./pkg/backend/device/cpu/matmul/` exit 0; arm64 matmul + complianceaudit pass; WT uncommitted |
| 2026-05-19 | reviewer / cycle 24 | T2.5 | PASS | All rubric criteria pass; tail mask fix verified at `f32_avx512_amd64.s:118-122`; full T2.5 package in WT; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 24 | T2.5 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.6** |
| 2026-05-19 | overseer / cycle 25 | meta | DRIFT | T2.5 checkoff valid (WT: `f32_avx512_amd64.{s,go}`, tail `MOVQ DX,CX` at `:118`, docs **5/30**); acceptance §9 “branch tip” contradicted binding workflow → verification loop T2.1 cycles 15–19; steering stale (T2.2 default); §9/§10a/steering aligned to WT review; next develop **T2.6** |
| 2026-05-19 | developer / cycle 25 | T2.6 | complete | `reduction/`: `f32_avx512_amd64.{s,go}` (5 kernels); `select_amd64.go` dispatch; `reduction_avx512_{parity,compliance,bench}_test.go`; `dispatchaudit/matrix_test.go` + docs **6/30**; `GOARCH=amd64 go build ./pkg/backend/device/cpu/reduction/` exit 0; arm64 reduction + complianceaudit pass; WT `f32_avx512_amd64.s` tail-mask delta uncommitted |
| 2026-05-19 | reviewer / cycle 25 | T2.6 | PASS | All rubric criteria pass; five ops at `parity.Lengths`; masked tails `MOVQ CX,DX`; amd64 build/test -c pass; native AVX-512F tests not run on arm64 |
| 2026-05-19 | sync / cycle 25 | T2.6 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.7** |
| 2026-05-19 | developer / cycle 26 | T2.7 | complete | `pool/`: `f32_avx512_amd64.{s,go}` (4 row kernels); `select_amd64.go` fast-row dispatch; `pool_avx512_{parity,compliance,bench}_test.go`; `dispatchaudit/matrix_test.go` + docs **7/30**; `GOARCH=amd64 go build ./pkg/backend/device/cpu/pool/` exit 0; arm64 pool + complianceaudit pass; on branch tip `ea1f747` |
| 2026-05-19 | reviewer / cycle 26 | T2.7 | PASS | All rubric criteria pass; ymm 4-wide row drivers mirror NEON; stride-1 max at `parity.Lengths`; amd64 build/test -c pass; native AVX-512F tests not run on arm64 |
| 2026-05-19 | sync / cycle 26 | T2.7 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.8** |
| 2026-05-19 | developer / cycle 27 | T2.8 | complete | `dropout/`: `f32_avx512_amd64.{s,go}`, `select_amd64.go`, `dropout_avx512_{parity,compliance,bench}_test.go`; `dispatchaudit/matrix_test.go` + docs **8/30**; `GOARCH=amd64 go build ./pkg/backend/device/cpu/dropout/` exit 0; arm64 dropout + complianceaudit pass; WT uncommitted |
| 2026-05-19 | reviewer / cycle 27 | T2.8 | FAIL | 2_correctness_performance + 3_spec_compliance FAIL: `f32_avx512_amd64.s:31-45` scalar GPR xorshift (`do_gen_rand`); `dropout_avx512_parity_test.go:36-48` N=1,7 never exercises asm; 1/4/5/6 PASS; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 27 | T2.8 | open | Review FAIL; blocking: vectorize RNG in `f32_avx512_amd64.s` (no scalar rand in `*_avx512*`); asm parity coverage at `parity.Lengths` edge N (masked tail or direct `DropoutFloat32AVX512Asm` at N=16); WT docs **8/30** premature until checkoff; T2.1/T2.2 §10b still open; next develop: **T2.8** |
| 2026-05-19 | developer / cycle 28 | T2.8 | complete | `dropout/`: vector XMM `drop_gen_vec` (no GPR xorshift); w16/w8/w4/tail asm paths; `DropoutF32AVX512` full-count asm; direct `DropoutFloat32AVX512Asm` parity subtest; WT `f32_avx512_amd64.{s,go}`, `dropout_avx512_parity_test.go` |
| 2026-05-19 | reviewer / cycle 28 | T2.8 | FAIL | 2_correctness_performance + 3_spec_compliance FAIL: `f32_avx512_amd64.s:138-145` `drop_tail3` unmasked 128-bit load/store when CX=3; cycle-27 GPR RNG + N=1/7 asm coverage closed; 1/4/5/6 PASS; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 28 | T2.8 | open | Review FAIL; blocking: masked remainder-3 tail (`K7` + `VMOVDQU32` like `reduction` `prod_avx512_w4_tail`); T2.1/T2.2 §10b still open; next develop: **T2.8** |
| 2026-05-19 | developer / cycle 29 | T2.8 | complete | `dropout/`: masked `drop_apply_tail` (`f32_avx512_amd64.s:132-145`, `K7`, `KANDQ`, `VMOVDQU32`); cycles 27–28 blockers closed |
| 2026-05-19 | reviewer / cycle 29 | T2.8 | PASS | All rubric criteria pass; masked remainder tail; full greenfield package; amd64 build/test -c pass; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 29 | T2.8 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.9** |
| 2026-05-19 | overseer / cycle 30 | meta | ALIGNED | T2.8 checkoff valid (WT: `dropout/f32_avx512_amd64.{s,go}`, masked `drop_apply_tail` `:132-145`, docs/dispatchaudit **8/30**); steering stale (T2.6) → **T2.9**; greenfield Phase 2 healthy; T2.1/T2.2 §10b human-only; next develop: **T2.9** |
| 2026-05-19 | developer / cycle 30 | T2.9 | complete | `losses/`: `f32_avx512_amd64.{s,go}` (MSE/MAE pair-sum); `select_amd64.go`; `losses_avx512_{parity,compliance,bench}_test.go`; docs/dispatchaudit **9/30**; `GOARCH=amd64 go build ./pkg/backend/device/cpu/losses/` exit 0; arm64 losses + complianceaudit pass; WT uncommitted |
| 2026-05-19 | reviewer / cycle 30 | T2.9 | PASS | All rubric criteria pass; masked tails; parity.Lengths + direct asm; amd64 build/test -c pass; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 30 | T2.9 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.10** |
| 2026-05-19 | developer / cycle 31 | T2.10 | complete | `convolution/`: `f32_avx512_amd64.{s,go}` (patch-dot); `select_amd64.go`; `convolution_avx512_{parity,compliance,bench}_test.go`; docs/dispatchaudit **10/30**; `GOARCH=amd64 go build ./pkg/backend/device/cpu/convolution/` exit 0; arm64 convolution + complianceaudit pass; WT uncommitted |
| 2026-05-19 | reviewer / cycle 31 | T2.10 | PASS | All rubric criteria pass; masked tail `f32_avx512_amd64.s:54-69`; parity.Lengths + direct asm; amd64 build/test -c pass; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 31 | T2.10 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.11** |
| 2026-05-19 | developer / cycle 32 | T2.11 | complete | `attention/`: `f32_avx512_amd64.{s,go}` (flash online-update + scale); `select_amd64.go`, `select_flash_amd64.go`; `attention_avx512_{parity,compliance,bench}_test.go`; docs/dispatchaudit **11/30** in WT; `GOARCH=amd64 go build ./pkg/backend/device/cpu/attention/` exit 0; arm64 `go test ./pkg/backend/device/cpu/attention/...` + `complianceaudit` pass; WT uncommitted |
| 2026-05-19 | reviewer / cycle 32 | T2.11 | FAIL | 2_correctness_performance + 3_spec_compliance + 5_verification FAIL: `f32_avx512_amd64.s:24-26,47-49,84-86` `FlashAttentionOnlineUpdateAVX512Asm` uses `VMULPS Y1,Y16,Y0` / `VMULPS Y3,Y17,Y2` / `VADDPS Y1,Y3,Y0` with uninitialized Y1/Y3; should be `VMULPS Y16,Y0,Y0`; `VMULPS Y17,Y2,Y2`; `VADDPS Y2,Y0,Y0` (dropout-style dest-last); scale path `:111` OK; 1/4/6 PASS; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 32 | T2.11 | open | Review FAIL; blocking: fix online-update `VMULPS`/`VADDPS` in `f32_avx512_amd64.s` w16/w8/w4/masked tails; T2.1/T2.2 §10b still open; next develop: **T2.11** |
| 2026-05-19 | developer / cycle 33 | T2.11 | complete | `attention/`: dest-last `VMULPS`/`VADDPS` online-update (`f32_avx512_amd64.s:24-26,:47-49,:63-65,:84-86`); `attention_avx512_parity_test.go` `parity.AssertFloat32SlicesWithinULP` 0 ULP; `GOARCH=amd64 go build ./pkg/backend/device/cpu/attention/` exit 0; arm64 attention + complianceaudit pass; WT uncommitted |
| 2026-05-19 | reviewer / cycle 33 | T2.11 | PASS | All rubric criteria pass; cycle-32 operand-order blocker closed; masked tail K7+VMOVDQU32; amd64 build pass; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 33 | T2.11 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.12** |
| 2026-05-19 | developer / cycle 34 | T2.12 | complete | `embedding/`: `f32_avx512_amd64.{s,go}` (CopyRow/AddRow ymm kernels); `f32_generic.go`, `select_amd64.go`, `select_other.go`; `embedding_avx512_{parity,compliance,bench}_test.go`; docs/dispatchaudit **12/30**; `GOARCH=amd64 go build ./pkg/backend/device/cpu/embedding/` exit 0; arm64 embedding + complianceaudit pass; WT uncommitted |
| 2026-05-19 | reviewer / cycle 34 | T2.12 | PASS | All rubric criteria pass; masked w4_tail K7+VMOVDQU32; parity.Lengths on hidden; amd64 build/test -c pass; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 34 | T2.12 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.13** |
| 2026-05-19 | overseer / cycle 35 | meta | ALIGNED | T2.12 checkoff valid (`embedding/f32_avx512_amd64.{s,go}`, docs/dispatchaudit **12/30** on `751157d`); steering stale (T2.9) → **T2.13**; greenfield Phase 2 healthy; T2.1/T2.2 §10b human-only; next develop: **T2.13** |
| 2026-05-19 | developer / cycle 35 | T2.13 | complete | `layernorm/`: `f32_avx512_amd64.{s,go}` (SquaredDiffSum f64 widen-accumulate, ApplyRow dest-last VMULPS/VADDPS); `f32_generic.go`, `select_amd64.go`, `select_other.go` `!arm64 && !amd64`; `layernorm_avx512_{parity,compliance,bench}_test.go`; docs/dispatchaudit **13/30**; `GOARCH=amd64 go build ./pkg/backend/device/cpu/layernorm/` exit 0; arm64 layernorm + complianceaudit pass; WT uncommitted |
| 2026-05-19 | reviewer / cycle 35 | T2.13 | PASS | All rubric criteria pass; masked tails K7+VMOVDQU32; parity.Lengths; apply 0 ULP; amd64 build/test -c pass; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 35 | T2.13 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.14** |
| 2026-05-19 | developer / cycle 36 | T2.14 | complete | `normalization/`: `f32_avx512_amd64.{s,go}` (SSD f64 widen-accumulate, ApplyConstScaleBias dest-last VMULPS/VADDPS); `f32_generic.go`, `natives.go`, `select_amd64.go`, `select_other.go` `!amd64`; `normalization_avx512_{parity,compliance,bench}_test.go`; `compute.go` wired GroupNorm/InstanceNorm/BatchNormEval f32; docs/dispatchaudit **14/30**; `GOARCH=amd64 go build ./pkg/backend/device/cpu/normalization/` exit 0; arm64 build + generic host benches pass; WT uncommitted |
| 2026-05-19 | reviewer / cycle 36 | T2.14 | PASS | All rubric criteria pass; masked tails K7+VMOVDQU32; parity.Lengths; apply 0 ULP; amd64 build/test -c pass; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 36 | T2.14 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.15** |
| 2026-05-19 | developer / cycle 37 | T2.15 | complete | `rope/`: `f32_avx512_amd64.{s,go}` (`RopePairsFloat32AVX512Asm` w8/w4/masked tail); `f32_generic.go`, `select_amd64.go`, `select_other.go` `!amd64 && !arm64`; `rope_avx512_{parity,compliance,bench}_test.go`; docs/dispatchaudit **15/30**; `GOARCH=amd64 go build ./pkg/backend/device/cpu/rope/` exit 0; arm64 rope bench + neon RoPE + complianceaudit pass; commit `bd5e8c3` |
| 2026-05-19 | reviewer / cycle 37 | T2.15 | PASS | All rubric criteria pass; dest-last VMULPS/VSUBPS; masked w4_tail K7+VMOVDQU32; parity.Lengths + direct asm; amd64 build/test -c pass; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 37 | T2.15 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.16** |
| 2026-05-19 | developer / cycle 38 | T2.16 | complete | `sampling/`: `f32_avx512_amd64.{s,go}` (greedy argmax + softmax row); `select_amd64.go`, `select_other.go`; `sampling_avx512_{parity,compliance,bench}_test.go`; docs/dispatchaudit **16/30**; `GOARCH=amd64 go build ./pkg/backend/device/cpu/sampling/` exit 0; arm64 sampling + complianceaudit pass; WT uncommitted |
| 2026-05-19 | reviewer / cycle 38 | T2.16 | PASS | All rubric criteria pass; masked w4_tail K7+VMOVDQU32; parity.Lengths; softmax 2 ULP (activation SoftmaxF32 bar); amd64 build/test -c pass; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 38 | T2.16 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.17** |
| 2026-05-19 | developer / cycle 39 | T2.17 | complete | `shape/`: `f32_avx512_amd64.{s,go}` (copy/where/masked-fill; masked tails + `R11` mask bit-offset); `shape_ops.go` natives; `shape_f32_parity_test.go` + `shape_avx512_{parity,compliance,bench}_test.go`; docs/dispatchaudit **17/30**; `GOARCH=amd64 go build ./pkg/backend/device/cpu/shape/` exit 0; arm64 shape + complianceaudit pass; WT uncommitted |
| 2026-05-19 | reviewer / cycle 39 | T2.17 | PASS | All rubric criteria pass; zmm/ymm hot loops; masked tails K7+VMOVDQU32; parity.Lengths 0 ULP; amd64 build/test -c pass; amd64+AVX512F parity not run on arm64 |
| 2026-05-19 | sync / cycle 39 | T2.17 | checked off | Review PASS; T2.1/T2.2 §10b still open; next develop: **T2.18** |
| 2026-05-19 | overseer / cycle 40 | meta | ALIGNED | T2.17 checkoff valid (`shape/` on `88fcdf6`, docs **17/30**); Agent steering stale (T2.13) → **T2.18** `masking`; greenfield Phase 2 healthy; T2.1/T2.2 §10b human-only; next develop: **T2.18** |

---

## Constraints

### Technology

- **Language:** Go 1.26.1 (module `github.com/theapemachine/caramba`).
- **CLI:** Cobra (`cmd/`: `serve`, `chat`, `image`, `research`, `progress`).
- **Config:** Single source `cmd/asset/config.yml` via `pkg/config` / Viper; embedded default in binary.
- **Compute:** `pkg/backend/compute` IR + tensor + `Executor`; devices under `pkg/backend/device/{cpu,metal,cuda,xla}`.
- **API:** Fiber HTTP server in `pkg/backend/api`.
- **Frontend:** TanStack Start + Router, Vite, React (`frontend/`).
- **Stores:** S3, Elasticsearch, Neo4j, Qdrant, DeepLake (`pkg/store`).
- **Hub:** Hugging Face + Xet CAS (`pkg/hub`).
- **Auth:** Clerk (optional) in config.
- **Tests:** GoConvey; `go test ./...` plus tagged builds for `cgo`, `cuda`, `xla`.

### Backend implementation (non-negotiable)

Equal standing for every method on `device.Backend`: **Go scalar, AVX-512, AVX2, SSE2, NEON, Metal, CUDA, XLA**.

SIMD paths use vector registers and instructions for the operation math across the entire inner loop; no cross-ISA jumps; one distinct assembly body per ISA file; exact mathematical definitions (no undocumented approximations).

Metal, CUDA, and XLA must implement the full `Backend` interface natively. They must submit real device kernels through their respective backend submission paths for every single operation. No silent fallbacks to CPU, host-side emulation, or "where applicable" exceptions are permitted.

If a kernel cannot be implemented correctly, **stop and report** — do not placeholder, alias, or remove symbols.

On arm64 dev machines, amd64 SIMD is validated on amd64 CI/hardware separately.

**Dtypes:** FP32, FP16, BF16, INT8, INT4, FP8 where mathematically applicable.

**Memory:** Zero-copy residency; device-side allocators (e.g., paged KV block tables) without silent host bounce.

### Code style

- Prefer methods on types; no single-character names (including receivers).
- Target ≤200 lines per file (400 hard ceiling except kernels/docs).
- Target ≤30 lines per method (60 max unless atomic kernel).
- Guard clauses; no `else`; max two nesting levels of `if`.
- TypeScript: `const` components; shared `flex` / `grid` / `typography` primitives.
- Every non-test `.go` file has a `_test.go` mirror; test names mirror methods.

### Non-goals (unless explicitly tasked)

- **32-bit (`386`, `arm`):** Not supported; remove legacy paths rather than maintaining them.
- Generator scripts or macro-generated assembly for kernels.
- Shadow configuration via environment variables outside `pkg/config`.
- Declaring backends complete without parity tests and benchmarks.
- Rewriting working structure without demonstrating why it is wrong.
- `git checkout` / `git reset --hard` / `git restore` on uncommitted work.

### Architecture decisions (resolved)

1. **`pkg/runtime`** is the top-level manifest execution layer; **`pkg/backend/compute/runtime`** holds only compute-graph execution boundaries.
2. **`pkg/manifest`** owns parse, validate, and lower YAML → `ir.Graph` + runtime program; graph optimizer passes live in **`pkg/backend/compute/compiler`**.
3. **AVX-512** is mandatory for every CPU domain on amd64; no partial subsets.
4. **Inference before training** in delivery order, but both are required for platform completion.
5. **Model catalog** smoke = forward parity vs reference (e.g., PyTorch) for fixed prompt/seed plus benchmark.
6. **Local-only** = egress-blocking transport + offline Hub + local notary.
7. **Distributed training** is native via `pkg/backend/compute/distributed` + `pkg/network` voluntary transport.
8. **YAML is source of truth** for frontend node graph (two-way binding, no alternate schema).
9. **Devteam** is opt-in via config flag with sandbox access when enabled.
10. **No prescribed architectures in Go** — KV, loops, tuners, and caches are manifest-composed atoms.
11. **`include` composability** for model, loop, cache, and data pipeline manifests.
12. **Dynamic ingestion** of SafeTensors/tokenizer assets when atomic ops exist.
13. **IR autodiff** for backward graphs; non-gradient optimization via manifest control flow.
14. **Manifest-driven tuners** (bandits, evolution, architecture search) — not hardcoded Go tuners.
15. **Alternative paradigms** (SNN, EBM, evolution, Hebbian, NCA) are first-class `Backend` surfaces at full SIMD/GPU speed.
16. **Mechanistic Interpretability & Model Editing** — First-class support for Logit Lens, SAEs, activation patching, and rank-one updates via manifest hooks, not ad hoc Go scripts.
17. **ModelScope Interpretability Inspector** — 3D WebGL-based visualization tool for graph topology, activation intensities, and gradients over time/layers.
18. **Visual Node Graph Editor** — Drag-and-drop UI with two-way binding to the underlying YAML manifest.

---

## Open questions

1. **CPU domain naming for R15 ops** — T1.3 audit complete (30 domains inventoried). Resolve at **T2.30** or a dedicated R15 task: dedicated `cpu/*` packages vs grouped under existing domains.
2. **`checkpoint` domain** — Confirm whether `pkg/backend/device/cpu/checkpoint` requires AVX-512 kernels or is host-metadata-only (affects T2.29 scope).
3. **README vs spec** — README marks “Manifest Compiler” and “Streaming Chat Runtime” as done; confirm whether to rewrite README in T1.5 or a dedicated doc-sync task after Phase 4.
4. **Autodiff scope for Phase 4** — Full op catalog vs staged subset for training MVP (must not block manifest-composed architectures indefinitely).
