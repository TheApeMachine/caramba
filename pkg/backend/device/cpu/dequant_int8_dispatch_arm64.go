//go:build arm64

package cpu

func DequantInt8Native(dst []float32, src []int8, scale float32, zeroPoint int8) {
	if len(dst) == 0 {
		return
	}

	DequantInt8NEONAsm(&dst[0], &src[0], len(dst), scale, int16(zeroPoint))
}
