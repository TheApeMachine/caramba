package convert

/*
SIMD entry-point stubs for F32↔F64. See bf16_f32_stubs.go for the
contract.
*/

func float32ToFloat64AVX512(dst []float64, src []float32) error {
	return Float32ToFloat64(dst, src)
}

func float64ToFloat32AVX512(dst []float32, src []float64) error {
	return Float64ToFloat32(dst, src)
}

func float32ToFloat64AVX2(dst []float64, src []float32) error {
	return Float32ToFloat64(dst, src)
}

func float64ToFloat32AVX2(dst []float32, src []float64) error {
	return Float64ToFloat32(dst, src)
}

func float32ToFloat64SSE2(dst []float64, src []float32) error {
	return Float32ToFloat64(dst, src)
}

func float64ToFloat32SSE2(dst []float32, src []float64) error {
	return Float64ToFloat32(dst, src)
}

func float32ToFloat64NEON(dst []float64, src []float32) error {
	return Float32ToFloat64(dst, src)
}

func float64ToFloat32NEON(dst []float32, src []float64) error {
	return Float64ToFloat32(dst, src)
}
