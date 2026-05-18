//go:build amd64

package convert

import (
	"github.com/theapemachine/caramba/pkg/dtype"
	"golang.org/x/sys/cpu"
)

/*
amd64 dispatchers for BF16↔F32. The dispatch order at call time is:

  1. AVX-512-BF16 (vcvtne2ps2bf16, vcvtbf162ps) — single-instruction
     hardware conversion when cpu.X86.HasAVX512 is set and the F16C
     / BF16 extensions are present. Detected at startup.
  2. AVX2 — manual shift/widen for bf16→f32; manual round-to-nearest-
     even truncation for f32→bf16.
  3. SSE2 — same approach as AVX2 with 128-bit vectors.
  4. Scalar fallback — the reference body in bf16_f32.go.

The .s files (bf16_f32_avx512_amd64.s, etc.) land in a hardware-
verified session; this dispatcher routes through the scalar body
until then. The Go-side surface pins the dispatch order so the SIMD
bodies drop in without changing call sites.
*/

var (
	hasAVX512 = cpu.X86.HasAVX512
	hasAVX2   = cpu.X86.HasAVX2
)

func init() {
	_, _ = hasAVX512, hasAVX2
}

func bfloat16ToFloat32(dst []float32, src []dtype.BF16) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	switch {
	case hasAVX512:
		return bfloat16ToFloat32AVX512(dst, src)
	case hasAVX2:
		return bfloat16ToFloat32AVX2(dst, src)
	}

	return bfloat16ToFloat32SSE2(dst, src)
}

func float32ToBFloat16(dst []dtype.BF16, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	switch {
	case hasAVX512:
		return float32ToBFloat16AVX512(dst, src)
	case hasAVX2:
		return float32ToBFloat16AVX2(dst, src)
	}

	return float32ToBFloat16SSE2(dst, src)
}
