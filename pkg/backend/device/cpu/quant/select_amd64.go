//go:build amd64

package quant

import "golang.org/x/sys/cpu"

func QuantInt8Native(dst []int8, src []float32, scale float32, zeroPoint int8) {
	if len(dst) == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		quantInt8AVX512(dst, src, scale, zeroPoint)

		return
	}

	quantInt8Generic(dst, src, scale, zeroPoint)
}
