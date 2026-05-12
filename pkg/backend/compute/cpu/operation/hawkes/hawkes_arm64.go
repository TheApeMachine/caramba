//go:build arm64

package hawkes

//go:noescape
func expSumNEON(expBuf []float64) float64

//go:noescape
func hawkesExcitationNEON(events []float64, now, beta, alpha float64) float64

//go:noescape
func hawkesKernelRowNEON(out, events []float64, ti, alpha, beta float64)

func hawkesExcitation(events []float64, now, beta, alpha float64) float64 {
	return hawkesExcitationNEON(events, now, beta, alpha)
}

func hawkesKernelRow(out, events []float64, ti, alpha, beta float64) {
	hawkesKernelRowNEON(out, events, ti, alpha, beta)
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
		if row > 0 && times[row] < times[row-1] {
			panic("hawkes: applyKernelMatrix: times must be non-decreasing")
		}
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
