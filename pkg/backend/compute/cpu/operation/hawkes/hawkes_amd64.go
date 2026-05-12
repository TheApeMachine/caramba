//go:build amd64

package hawkes

import (
	"math"

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
func expSumAVX2(expBuf []float64) float64

//go:noescape
func expSumSSE2(expBuf []float64) float64

//go:noescape
func subVecAVX2(dst, a, b []float64)

//go:noescape
func subVecSSE2(dst, a, b []float64)

func applyIntensity(out, times, alpha, beta, mu []float64, t float64, K, T int) {
	cutoff := 0
	for cutoff < T && times[cutoff] < t {
		cutoff++
	}

	validTimes := times[:cutoff]
	n := len(validTimes)

	if n == 0 {
		for k := 0; k < K; k++ {
			out[k] = mu[k]
		}
		return
	}

	expBuf := make([]float64, n)

	for k := 0; k < K; k++ {
		bk := beta[k]
		for i := 0; i < n; i++ {
			expBuf[i] = math.Exp(-bk * (t - validTimes[i]))
		}

		var sum float64
		if useAVX2 {
			sum = expSumAVX2(expBuf)
		} else {
			sum = expSumSSE2(expBuf)
		}

		out[k] = mu[k] + alpha[k]*sum
	}
}

func applyKernelMatrix(out, times []float64, alpha, beta float64, T int) {
	if T <= 0 || len(times) < T || len(out) < T*T {
		panic("hawkes: applyKernelMatrix: need T > 0, len(times) >= T, len(out) >= T*T")
	}

	tmp := make([]float64, T)

	for row := 0; row < T; row++ {
		ti := times[row]
		rowSlice := times[row+1:]
		rowLen := T - row - 1

		if rowLen <= 0 {
			continue
		}

		tmpSlice := tmp[:rowLen]

		for idx := 0; idx < rowLen; idx++ {
			tmpSlice[idx] = rowSlice[idx] - ti
		}

		for idx := 0; idx < rowLen; idx++ {
			out[row*T+(row+1+idx)] = alpha * math.Exp(-beta*tmpSlice[idx])
		}
	}
}

func alignedLen(n int) int {
	width := 2

	if useAVX2 {
		width = 4
	}

	return n / width * width
}
