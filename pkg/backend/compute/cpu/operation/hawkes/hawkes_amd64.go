//go:build amd64

package hawkes

import (
	"golang.org/x/sys/cpu"

	mathops "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
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
	expOut := make([]float64, T)

	for row := 0; row < T; row++ {
		ti := times[row]
		rowSlice := times[row+1:]
		rowLen := T - row - 1

		if rowLen <= 0 {
			continue
		}

		tmpSlice := tmp[:rowLen]
		expSlice := expOut[:rowLen]
		copy(tmpSlice, rowSlice[:rowLen])
		mathops.AddScalarVec(tmpSlice, -ti)
		mathops.ScaleVec(tmpSlice, -beta)
		mathops.ExpVec(expSlice, tmpSlice)
		mathops.ScaleVec(expSlice, alpha)
		copy(out[row*T+row+1:row*T+row+1+rowLen], expSlice)
	}
}

func alignedLen(n int) int {
	width := 2

	if useAVX2 {
		width = 4
	}

	return n / width * width
}
