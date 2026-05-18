/*
Package convert is the SIMD conversion kernel home. Every
dtype↔dtype conversion is a first-class kernel here with the same
five-host-ISA expectation as every other SIMD operation per AGENTS.md
§1: scalar Go (reference) + AVX-512 + AVX2 + SSE2 + NEON, plus
Metal / CUDA / XLA paths when they apply.

Per the spray-and-pray contract (VERIFICATION_STATUS.md), this
package's scalar bodies are correct and tested. The SIMD variants
exist as stub bodies that fall through to the scalar reference; their
real assembly land in later sessions where they can be benchmarked
on real amd64 / arm64 hosts.

The public surface here is the same shape as pkg/dtype/convert
(BytesToFloat64, etc.), but operates on caller-owned slices rather
than allocating per call, so kernels in the dispatch tables can reuse
output buffers across forward passes.

Coverage of dtype pairs:

  - bf16 ↔ f32 (f32 ↔ bf16 conversion is the most heavily used path)
  - f16 ↔ f32
  - f32 ↔ f64
  - bf16 ↔ f16 (via f32 internally)
  - fp8e4m3 ↔ f32, fp8e5m2 ↔ f32
  - int8 ↔ f32 (dequant only; the parameterized quantization kernel
    that takes a scale factor lives in pkg/backend/compute/kernels).
  - int4 ↔ f32 (dequant)
*/
package convert
