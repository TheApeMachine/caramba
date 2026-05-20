# CPU dispatch matrix (T1.3)

Per-domain registration of scalar (Go reference) and SIMD paths (AVX-512, AVX2, SSE2 on amd64; NEON on arm64). **registered** means at least one assembly file or dispatch-table entry exists in that domain's package; it does not assert full operation coverage.

Machine-checkable source: `pkg/backend/device/cpu/dispatchaudit/`, validated by `dispatchaudit_test.go`.

Device backends (T1.4): [`docs/device-backend-matrix.md`](./device-backend-matrix.md).

Combined coverage (T1.5): [`backend-coverage.md`](./backend-coverage.md).

Alternative paradigms (T2.30): [`cpu-alternative-paradigm-avx512.md`](./cpu-alternative-paradigm-avx512.md).

## Summary

| ISA path | Domains registered |
|----------|-------------------:|
| Scalar (Go) | 32 / 32 |
| AVX-512 (amd64) | 32 / 32 |
| AVX2 (amd64) | 2 / 32 |
| SSE2 (amd64) | 2 / 32 |
| NEON (arm64) | 20 / 32 |

## Per-domain matrix

| Domain | Scalar | AVX-512 | AVX2 | SSE2 | NEON |
|--------|:------:|:-------:|:----:|:----:|:----:|
| activation | yes | yes | yes | yes | yes |
| active_inference | yes | yes | — | — | — |
| attention | yes | yes | — | — | yes |
| causal | yes | yes | — | — | yes |
| checkpoint | yes | yes | — | — | — |
| convolution | yes | yes | — | — | yes |
| dequant | yes | yes | — | — | yes |
| dot | yes | yes | — | — | yes |
| dropout | yes | yes | — | — | yes |
| elementwise | yes | yes | — | — | yes |
| embedding | yes | yes | — | — | — |
| hawkes | yes | yes | — | — | yes |
| interpretability | yes | yes | — | — | — |
| layernorm | yes | yes | — | — | yes |
| losses | yes | yes | — | — | yes |
| masking | yes | yes | — | — | — |
| math | yes | yes | — | — | — |
| matmul | yes | yes | — | — | yes |
| model_editing | yes | yes | — | — | — |
| normalization | yes | yes | — | — | — |
| optimizer | yes | yes | — | — | yes |
| physics | yes | yes | — | — | yes |
| pool | yes | yes | — | — | yes |
| pospop | yes | yes | yes | yes | yes |
| predictive_coding | yes | yes | — | — | — |
| quant | yes | yes | — | — | yes |
| reduction | yes | yes | — | — | yes |
| rope | yes | yes | — | — | yes |
| sampling | yes | yes | — | — | — |
| shape | yes | yes | — | — | — |
| tokenizer | yes | yes | — | — | — |
| vsa | yes | yes | — | — | yes |

### AVX-512 registered domains

- `checkpoint`
- `activation`
- `attention`
- `causal`
- `convolution`
- `dequant`
- `dot`
- `dropout`
- `elementwise`
- `embedding`
- `hawkes`
- `interpretability`
- `layernorm`
- `losses`
- `masking`
- `math`
- `matmul`
- `model_editing`
- `normalization`
- `optimizer`
- `physics`
- `pool`
- `pospop`
- `quant`
- `reduction`
- `rope`
- `sampling`
- `shape`
- `tokenizer`
- `predictive_coding`
- `vsa`

## Registration rules

1. **Scalar** — `select_generic.go`, `*_generic.go`, or dispatch tables listing `"generic"` / `*Generic` in the domain package.
2. **AVX-512 / AVX2 / SSE2** — `*avx512*`, `*avx2*`, or `*sse2*` assembly under the domain, and/or `select_amd64.go` (or amd64 select shards) declaring matching symbols or `"avx512"` / `"avx2"` / `"sse2"` dispatch names.
3. **NEON** — `*_neon_arm64.s` or `select_arm64.go` with `NEON` symbols or `"neon"` dispatch names.
4. **`cpu/neon`** — shared ARM64 helpers; excluded from this domain table.
