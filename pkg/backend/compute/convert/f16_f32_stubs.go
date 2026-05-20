package convert

import "github.com/theapemachine/manifesto/dtype"

/*
SIMD entry-point stubs for F16↔F32. See bf16_f32_stubs.go for the
contract. Real F16C-based AVX2/AVX-512 paths plus NEON fcvt/fcvtn
implementations land in later sessions.
*/

func float16ToFloat32AVX512(dst []float32, src []dtype.F16) error {
	return Float16ToFloat32(dst, src)
}

func float32ToFloat16AVX512(dst []dtype.F16, src []float32) error {
	return Float32ToFloat16(dst, src)
}

func float16ToFloat32AVX2(dst []float32, src []dtype.F16) error {
	return Float16ToFloat32(dst, src)
}

func float32ToFloat16AVX2(dst []dtype.F16, src []float32) error {
	return Float32ToFloat16(dst, src)
}

func float16ToFloat32SSE2(dst []float32, src []dtype.F16) error {
	return Float16ToFloat32(dst, src)
}

func float32ToFloat16SSE2(dst []dtype.F16, src []float32) error {
	return Float32ToFloat16(dst, src)
}

func float16ToFloat32NEON(dst []float32, src []dtype.F16) error {
	return Float16ToFloat32(dst, src)
}

func float32ToFloat16NEON(dst []dtype.F16, src []float32) error {
	return Float32ToFloat16(dst, src)
}
