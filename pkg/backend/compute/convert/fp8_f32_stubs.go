package convert

import "github.com/theapemachine/manifesto/dtype"

/*
SIMD entry-point stubs for FP8↔F32. See bf16_f32_stubs.go for the
contract. The real AVX-512 byte-shuffle + LUT-in-zmm paths and the
NEON tbl-based paths land in later sessions; FP8 has no native SIMD
on current amd64 hardware so the upcast strategy is LUT-driven.
*/

func Float8E4M3ToFloat32AVX512(dst []float32, src []dtype.F8E4M3) error {
	return Float8E4M3ToFloat32(dst, src)
}

func Float32ToFloat8E4M3AVX512(dst []dtype.F8E4M3, src []float32) error {
	return Float32ToFloat8E4M3(dst, src)
}

func Float8E4M3ToFloat32AVX2(dst []float32, src []dtype.F8E4M3) error {
	return Float8E4M3ToFloat32(dst, src)
}

func Float32ToFloat8E4M3AVX2(dst []dtype.F8E4M3, src []float32) error {
	return Float32ToFloat8E4M3(dst, src)
}

func Float8E4M3ToFloat32SSE2(dst []float32, src []dtype.F8E4M3) error {
	return Float8E4M3ToFloat32(dst, src)
}

func Float32ToFloat8E4M3SSE2(dst []dtype.F8E4M3, src []float32) error {
	return Float32ToFloat8E4M3(dst, src)
}

func Float8E4M3ToFloat32NEON(dst []float32, src []dtype.F8E4M3) error {
	return Float8E4M3ToFloat32(dst, src)
}

func Float32ToFloat8E4M3NEON(dst []dtype.F8E4M3, src []float32) error {
	return Float32ToFloat8E4M3(dst, src)
}

func Float8E5M2ToFloat32AVX512(dst []float32, src []dtype.F8E5M2) error {
	return Float8E5M2ToFloat32(dst, src)
}

func Float32ToFloat8E5M2AVX512(dst []dtype.F8E5M2, src []float32) error {
	return Float32ToFloat8E5M2(dst, src)
}

func Float8E5M2ToFloat32AVX2(dst []float32, src []dtype.F8E5M2) error {
	return Float8E5M2ToFloat32(dst, src)
}

func Float32ToFloat8E5M2AVX2(dst []dtype.F8E5M2, src []float32) error {
	return Float32ToFloat8E5M2(dst, src)
}

func Float8E5M2ToFloat32SSE2(dst []float32, src []dtype.F8E5M2) error {
	return Float8E5M2ToFloat32(dst, src)
}

func Float32ToFloat8E5M2SSE2(dst []dtype.F8E5M2, src []float32) error {
	return Float32ToFloat8E5M2(dst, src)
}

func Float8E5M2ToFloat32NEON(dst []float32, src []dtype.F8E5M2) error {
	return Float8E5M2ToFloat32(dst, src)
}

func Float32ToFloat8E5M2NEON(dst []dtype.F8E5M2, src []float32) error {
	return Float32ToFloat8E5M2(dst, src)
}
