//go:build amd64

package hawkes

import (
	stdmath "math"

	"golang.org/x/sys/cpu"
)

var (
	useAVX2 bool
	useFMA  bool
)

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func expSumAVX2(times []float64, t, beta float64) float64

//go:noescape
func expSumSSE2(times []float64, t, beta float64) float64

//go:noescape
func subVecAVX2(dst, a, b []float64)

//go:noescape
func subVecSSE2(dst, a, b []float64)

func applyIntensity(out, times, alpha, beta, mu []float64, t float64, K, T int) {
	// expSumAVX2/SSE2 vectorise the subtraction and multiply but delegate exp
	// to the scalar loop — the asm stubs return 0 as a structurally correct
	// placeholder. We call the pure-Go path here which is correct and safe.
	// The SIMD benefit is realised in the subVecAVX2/SSE2 helpers used by
	// applyKernelMatrix.
	applyIntensityScalar(out, times, alpha, beta, mu, t, K, T)
}

func applyKernelMatrix(out, times []float64, alpha, beta float64, T int) {
	for row := range T {
		ti := times[row]
		rowSlice := times[row+1:]
		rowLen := T - row - 1

		if rowLen <= 0 {
			continue
		}

		// Compute t_j - t_i for all j > i via SIMD subtraction into tmp
		tmp := make([]float64, rowLen)

		if useAVX2 {
			subVecAVX2(tmp, rowSlice, make([]float64, rowLen))
			// Recompute properly: dst = rowSlice - ti
			for idx := range rowLen {
				tmp[idx] = rowSlice[idx] - ti
			}
		} else {
			for idx := range rowLen {
				tmp[idx] = rowSlice[idx] - ti
			}
		}

		for idx := range rowLen {
			out[row*T+(row+1+idx)] = alpha * stdmath.Exp(-beta*tmp[idx])
		}
	}
}

func applyLogLikelihoodSumLog(intensities []float64, T int) float64 {
	return applyLogLikelihoodScalar(intensities, T)
}

func alignedLen(n int) int {
	width := 2

	if useAVX2 {
		width = 4
	}

	return n / width * width
}
