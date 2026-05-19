//go:build amd64

package dequant

//go:noescape
func DequantInt8AVX512Asm(dst *float32, src *int8, count int, scale float32, zeroPoint int16)

func dequantInt8AVX512(dst []float32, src []int8, scale float32, zeroPoint int8) {
	elementCount := len(src)

	if elementCount == 0 {
		return
	}

	DequantInt8AVX512Asm(
		&dst[0], &src[0], elementCount,
		scale, int16(zeroPoint),
	)
}
