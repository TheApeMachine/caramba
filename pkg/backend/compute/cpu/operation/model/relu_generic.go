//go:build !amd64 && !arm64

package model

// reluInPlace applies ReLU in-place: x[i] = max(0, x[i]).
func reluInPlace(x []float64) {
	for idx := range x {
		if x[idx] < 0 {
			x[idx] = 0
		}
	}
}
