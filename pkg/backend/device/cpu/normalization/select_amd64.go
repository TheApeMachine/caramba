//go:build amd64

package normalization

import "golang.org/x/sys/cpu"

func NormSquaredDiffSumNative(row []float32, mean float32) float32 {
	if cpu.X86.HasAVX512F {
		return normSquaredDiffSumF32AVX512(row, mean)
	}

	return NormSquaredDiffSumGeneric(row, mean)
}

func NormApplyConstScaleBiasNative(
	outRow, row []float32,
	mean, invStdDev, scale, bias float32,
) {
	if cpu.X86.HasAVX512F {
		normApplyConstScaleBiasF32AVX512(outRow, row, mean, invStdDev, scale, bias)

		return
	}

	NormApplyConstScaleBiasGeneric(outRow, row, mean, invStdDev, scale, bias)
}
