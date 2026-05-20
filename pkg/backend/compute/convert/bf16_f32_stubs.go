package convert

import "github.com/theapemachine/manifesto/dtype"

/*
SIMD entry-point stubs for the BF16↔F32 conversion kernels. Per
AGENTS.md §1 and the package doc, every conversion kernel must ship
scalar Go + AVX-512 + AVX2 + SSE2 + NEON variants. These stubs make
the symbols exist with the contracted signatures and fall through to
the scalar reference body in bf16_f32.go; the real ISA assembly land
in later sessions where they can be benchmarked on real hardware.
The stubs preserve return values and error semantics exactly.
*/

func bfloat16ToFloat32AVX512(dst []float32, src []dtype.BF16) error {
	return bfloat16ToFloat32Scalar(dst, src)
}

func float32ToBFloat16AVX512(dst []dtype.BF16, src []float32) error {
	return float32ToBFloat16Scalar(dst, src)
}

func bfloat16ToFloat32AVX2(dst []float32, src []dtype.BF16) error {
	return bfloat16ToFloat32Scalar(dst, src)
}

func float32ToBFloat16AVX2(dst []dtype.BF16, src []float32) error {
	return float32ToBFloat16Scalar(dst, src)
}

func bfloat16ToFloat32SSE2(dst []float32, src []dtype.BF16) error {
	return bfloat16ToFloat32Scalar(dst, src)
}

func float32ToBFloat16SSE2(dst []dtype.BF16, src []float32) error {
	return float32ToBFloat16Scalar(dst, src)
}

func bfloat16ToFloat32NEON(dst []float32, src []dtype.BF16) error {
	return bfloat16ToFloat32Scalar(dst, src)
}

func float32ToBFloat16NEON(dst []dtype.BF16, src []float32) error {
	return float32ToBFloat16Scalar(dst, src)
}
