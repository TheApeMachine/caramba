//go:build !amd64 && !arm64

package dequant

func DequantInt8Native(dst []float32, src []int8, scale float32, zeroPoint int8) {
	dequantInt8Generic(dst, src, scale, zeroPoint)
}
