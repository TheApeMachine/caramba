//go:build !amd64 && !arm64

package quant

func QuantInt8Native(dst []int8, src []float32, scale float32, zeroPoint int8) {
	quantInt8Generic(dst, src, scale, zeroPoint)
}
