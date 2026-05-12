//go:build amd64

package hawkes

import (
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

//go:noescape
func hawkesExcitationAVX2(events []float64, now, beta, alpha float64) float64

//go:noescape
func hawkesExcitationSSE2(events []float64, now, beta, alpha float64) float64

//go:noescape
func hawkesKernelRowAVX2(out, events []float64, ti, alpha, beta float64)

//go:noescape
func hawkesKernelRowSSE2(out, events []float64, ti, alpha, beta float64)

func hawkesExcitation(events []float64, now, beta, alpha float64) float64 {
	if useAVX2 && useFMA {
		return hawkesExcitationAVX2(events, now, beta, alpha)
	}

	return hawkesExcitationSSE2(events, now, beta, alpha)
}

func hawkesKernelRow(out, events []float64, ti, alpha, beta float64) {
	if useAVX2 && useFMA {
		hawkesKernelRowAVX2(out, events, ti, alpha, beta)
		return
	}

	hawkesKernelRowSSE2(out, events, ti, alpha, beta)
}

func applyIntensity(out, times, alpha, beta, mu []float64, t float64, K, T int) {
	cutoff := 0

	for cutoff < T && times[cutoff] < t {
		cutoff++
	}

	validTimes := times[:cutoff]

	if len(validTimes) == 0 {
		copy(out[:K], mu[:K])
		return
	}

	for k := 0; k < K; k++ {
		out[k] = mu[k] + hawkesExcitation(validTimes, t, beta[k], alpha[k])
	}
}

func applyKernelMatrix(out, times []float64, alpha, beta float64, T int) {
	if T <= 0 || len(times) < T || len(out) < T*T {
		panic("hawkes: applyKernelMatrix: need T > 0, len(times) >= T, len(out) >= T*T")
	}

	for row := 0; row < T; row++ {
		rowLen := T - row - 1

		if rowLen <= 0 {
			continue
		}

		hawkesKernelRow(
			out[row*T+row+1:row*T+row+1+rowLen],
			times[row+1:row+1+rowLen],
			times[row], alpha, beta,
		)
	}
}

func alignedLen(n int) int {
	width := 2

	if useAVX2 {
		width = 4
	}

	return n / width * width
}
