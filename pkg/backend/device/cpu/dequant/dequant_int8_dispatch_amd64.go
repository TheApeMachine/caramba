//go:build amd64

package dequant

import "golang.org/x/sys/cpu"

func DequantInt8Native(dst []float32, src []int8, scale float32, zeroPoint int8) {
	if len(src) == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		dequantInt8AVX512(dst, src, scale, zeroPoint)

		return
	}

	dequantInt8Generic(dst, src, scale, zeroPoint)
}
