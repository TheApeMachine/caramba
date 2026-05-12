//go:build arm64

package hawkes

import (
	mathops "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

//go:noescape
func expSumNEON(expBuf []float64) float64

func applyIntensity(out, times, alpha, beta, mu []float64, t float64, K, T int) {
	cutoff := 0

	for cutoff < T && times[cutoff] < t {
		cutoff++
	}

	validTimes := times[:cutoff]
	n := len(validTimes)

	if n == 0 {
		copy(out[:K], mu[:K])
		return
	}

	expBuf := make([]float64, n)
	scratch := make([]float64, n)

	for k := 0; k < K; k++ {
		bk := beta[k]

		for i := 0; i < n; i++ {
			scratch[i] = t - validTimes[i]
		}

		mathops.ScaleVec(scratch, -bk)
		mathops.ExpVec(expBuf, scratch)

		out[k] = mu[k] + alpha[k]*expSumNEON(expBuf)
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

	tmp := make([]float64, T)
	expOut := make([]float64, T)

	for row := 0; row < T; row++ {
		ti := times[row]
		rowLen := T - row - 1

		if rowLen <= 0 {
			continue
		}

		tmpSlice := tmp[:rowLen]
		expSlice := expOut[:rowLen]
		copy(tmpSlice, times[row+1:row+1+rowLen])
		mathops.AddScalarVec(tmpSlice, -ti)
		mathops.ScaleVec(tmpSlice, -beta)
		mathops.ExpVec(expSlice, tmpSlice)
		mathops.ScaleVec(expSlice, alpha)
		copy(out[row*T+row+1:row*T+row+1+rowLen], expSlice)
	}
}
