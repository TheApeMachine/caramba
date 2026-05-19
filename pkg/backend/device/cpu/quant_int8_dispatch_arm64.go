//go:build arm64

package cpu

func QuantInt8Native(dst []int8, src []float32, scale float32, zeroPoint int8) {
	if len(dst) == 0 {
		return
	}

	QuantInt8NEONAsm(&dst[0], &src[0], len(dst), 1.0/scale, int32(zeroPoint))
}
