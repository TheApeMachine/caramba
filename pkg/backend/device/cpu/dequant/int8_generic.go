package dequant

func dequantInt8Generic(dst []float32, src []int8, scale float32, zeroPoint int8) {
	for index := range src {
		dst[index] = float32(int32(src[index])-int32(zeroPoint)) * scale
	}
}
