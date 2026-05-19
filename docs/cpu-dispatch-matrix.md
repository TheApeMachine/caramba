# CPU dispatch matrix (T1.3)

Per-domain registration of scalar (Go reference) and SIMD paths (AVX-512, AVX2, SSE2 on amd64; NEON on arm64). **registered** means at least one assembly file or dispatch-table entry exists in that domain's package; it does not assert full operation coverage.

Machine-checkable source: `pkg/backend/device/cpu/dispatchaudit/`, validated by `dispatchaudit_test.go`.

Device backends (T1.4): [`docs/device-backend-matrix.md`](./device-backend-matrix.md).

Combined coverage (T1.5): [`backend-coverage.md`](./backend-coverage.md).

## Summary

| ISA path | Domains registered |
|----------|-------------------:|
| Scalar (Go) | 30 / 30 |
| AVX-512 (amd64) | 12 / 30 |
| AVX2 (amd64) | 2 / 30 |
| SSE2 (amd64) | 2 / 30 |
| NEON (arm64) | 20 / 30 |

## Per-domain matrix

| Domain | Scalar | AVX-512 | AVX2 | SSE2 | NEON |
|--------|:------:|:-------:|:----:|:----:|:----:|
| activation | yes | yes | yes | yes | yes |
| active_inference | yes | — | — | — | — |
| attention | yes | yes | — | — | yes |
| causal | yes | — | — | — | yes |
| checkpoint | yes | — | — | — | — |
| convolution | yes | yes | — | — | yes |
| dequant | yes | — | — | — | yes |
| dot | yes | yes | — | — | yes |
| dropout | yes | yes | — | — | yes |
| elementwise | yes | yes | — | — | yes |
| embedding | yes | yes | — | — | — |
| hawkes | yes | — | — | — | yes |
| layernorm | yes | — | — | — | yes |
| losses | yes | yes | — | — | yes |
| masking | yes | — | — | — | — |
| math | yes | — | — | — | — |
| matmul | yes | yes | — | — | yes |
| normalization | yes | — | — | — | — |
| optimizer | yes | — | — | — | yes |
| physics | yes | — | — | — | yes |
| pool | yes | yes | — | — | yes |
| pospop | yes | yes | yes | yes | yes |
| predictive_coding | yes | — | — | — | — |
| quant | yes | — | — | — | yes |
| reduction | yes | yes | — | — | yes |
| rope | yes | — | — | — | yes |
| sampling | yes | — | — | — | — |
| shape | yes | — | — | — | — |
| tokenizer | yes | — | — | — | — |
| vsa | yes | — | — | — | yes |

### AVX-512 registered domains

- `activation`
- `attention`
- `convolution`
- `dot`
- `dropout`
- `elementwise`
- `embedding`
- `losses`
- `matmul`
- `pool`
- `pospop`
- `reduction`

## Registration rules

1. **Scalar** — `select_generic.go`, `*_generic.go`, or dispatch tables listing `"generic"` / `*Generic` in the domain package.
2. **AVX-512 / AVX2 / SSE2** — `*avx512*`, `*avx2*`, or `*sse2*` assembly under the domain, and/or `select_amd64.go` (or amd64 select shards) declaring matching symbols or `"avx512"` / `"avx2"` / `"sse2"` dispatch names.
3. **NEON** — `*_neon_arm64.s` or `select_arm64.go` with `NEON` symbols or `"neon"` dispatch names.
4. **`cpu/neon`** — shared ARM64 helpers; excluded from this domain table.
