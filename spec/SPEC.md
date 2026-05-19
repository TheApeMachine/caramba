# Project spec

## Vision

**Caramba** is a manifest-driven machine learning research substrate: researchers declare topology, runtime behavior, state, and instrumentation in YAML (“a manifest is a model”). The manifest serves as a rudimentary programming language—atomic operations for control flow (e.g., looping for next-token generation), IO (e.g., stdin), and custom state—so the platform does not pre-prescribe KV caches, training loops, or sampling strategies in Go. Researchers compose those mechanisms in YAML and execute them with resident tensors on every supported compute target under strict accuracy and provenance contracts.

**What exists today**

- **Low-level compute:** `pkg/backend/device/cpu` — 30 operation domains (`activation` … `vsa`, excluding shared `cpu/neon`) with Go scalar on all; **NEON** registered in 20/30 domains; **amd64 AVX2/SSE2/AVX-512** assembly registered only in `activation` and `pospop` (per T1.3 audit). Legacy **386** paths removed; CPU targets **amd64** and **arm64** only (`GOARCH=386` unsupported).
- **Device backends:** `pkg/backend/device/{metal,cuda,xla}` — **Metal** has 462 `kernels.Default` registrations (158 unique names, 68/119 required ops); **CUDA** and **XLA** are tensor upload/download only (0 kernel registrations). **`device.Backend` ↔ `ir.RequiredOperationIDs()` inventory** in `docs/backend-inventory.md` and `pkg/backend/device/inventory*.go` (151 methods, 119 required ops cross-linked). **CPU dispatch matrix** in `docs/cpu-dispatch-matrix.md` and `pkg/backend/device/cpu/dispatchaudit/`. **Device backend matrix** in `docs/device-backend-matrix.md` and `pkg/backend/device/backendaudit/`. **Combined backend coverage** in `docs/backend-coverage.md` and `pkg/backend/device/coverageaudit/` (T1.2–T1.4 registration snapshot, R1 execution-target summary).
- **Compute core:** `pkg/backend/compute/{ir,tensor,kernels,fusion,state}`; minimal `pkg/backend/compute/runtime` (`executor.go`, `host.go` only).
- **Assets and templates:** `pkg/asset` (embedded YAML, `system.topology` shapes, `TemplateFS` for includes); model manifests under `pkg/asset/template/manifest/`; canonical runtime YAML in `pkg/asset/template/runtime/{chat,diffusion}.yml`.
- **Hub, provenance, API, frontend:** `pkg/hub`, `pkg/notary`, `pkg/backend/api` + `cmd/serve`, TanStack Start `frontend/` (node-graph, research, benchmark routes).
- **CLIs:** `cmd/chat` and `cmd/image` are **stubs** (`RunE` returns `nil`).

**Gaps (relative to README and `docs/research-platform-requirements.md`)**

- No `pkg/manifest` or `pkg/runtime` packages yet (docs describe target layout; code does not).
- No manifest→`ir.Graph` / runtime-program compiler pipeline wired end-to-end.
- No graph optimizer package at `pkg/backend/compute/compiler` (README checklist item is aspirational).
- **AVX-512** incomplete across CPU domains; **backend legality** and **zero-copy residency** tests not implemented.
- Training/fine-tuning, manifest-driven tuners, distributed/voluntary compute, and research UX (ModelScope, layer surgery, node-graph round-trip) are not done.
- Direct `os.Getenv` / `os.LookupEnv` remain in some `pkg/store/*` clients (violates R9 until migrated).

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
- [ ] T1.6 — Fix audit findings: ISA aliasing, scalar-in-SIMD files, widened test epsilons, forbidden comment phrasing (requirement: R1, R2, R10)
- [ ] T1.7 — Migrate `pkg/store` clients off direct `os.Getenv` / `os.LookupEnv` into `pkg/config` (requirement: R9)

### Phase 2: AVX-512 CPU kernels (amd64)

One task per domain: real `_avx512_amd64.s` bodies, dispatch wiring, parity at N ∈ {1, 7, 64, 1024, 8192}, benchmark. Skip domains with no amd64 SIMD today only if the audit (T1.3) marks them scalar-only by design.

- [ ] T2.1 — `activation` AVX-512: close gaps vs audit (requirement: R1, R2)
- [ ] T2.2 — `pospop` AVX-512 (requirement: R1, R2)
- [ ] T2.3 — `elementwise` AVX-512 (requirement: R1, R2)
- [ ] T2.4 — `dot` AVX-512 (requirement: R1, R2)
- [ ] T2.5 — `matmul` AVX-512 (requirement: R1, R2)
- [ ] T2.6 — `reduction` AVX-512 (requirement: R1, R2)
- [ ] T2.7 — `pool` AVX-512 (requirement: R1, R2)
- [ ] T2.8 — `dropout` AVX-512 (requirement: R1, R2)
- [ ] T2.9 — `losses` AVX-512 (requirement: R1, R2)
- [ ] T2.10 — `convolution` AVX-512 (requirement: R1, R2)
- [ ] T2.11 — `attention` AVX-512 (requirement: R1, R2)
- [ ] T2.12 — `embedding` AVX-512 (requirement: R1, R2)
- [ ] T2.13 — `layernorm` AVX-512 (requirement: R1, R2)
- [ ] T2.14 — `normalization` AVX-512 (requirement: R1, R2)
- [ ] T2.15 — `rope` AVX-512 (requirement: R1, R2)
- [ ] T2.16 — `sampling` AVX-512 (requirement: R1, R2)
- [ ] T2.17 — `shape` AVX-512 (requirement: R1, R2)
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
- [ ] T1.6 — Fix audit findings: ISA aliasing, scalar-in-SIMD files, widened test epsilons, forbidden comment phrasing
- [ ] T1.7 — Migrate `pkg/store` clients off direct `os.Getenv` / `os.LookupEnv` into `pkg/config`

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

1. **CPU domain naming for R15 ops** — Should spiking/EBM/evolutionary/Hebbian/NCA/interpretability live in dedicated `pkg/backend/device/cpu/*` packages or stay grouped under existing domains until T1.3 audit completes?
2. **`checkpoint` domain** — Confirm whether `pkg/backend/device/cpu/checkpoint` requires AVX-512 kernels or is host-metadata-only (affects T2.29 scope).
3. **README vs spec** — README marks “Manifest Compiler” and “Streaming Chat Runtime” as done; confirm whether to rewrite README in T1.5 or a dedicated doc-sync task after Phase 4.
4. **Autodiff scope for Phase 4** — Full op catalog vs staged subset for training MVP (must not block manifest-composed architectures indefinitely).
