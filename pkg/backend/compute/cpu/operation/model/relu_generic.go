//go:build !amd64 && !arm64

package model

func reluInPlace(x []float64) {
	for idx := range x {
		if x[idx] < 0 {
			x[idx] = 0
		}
	}
}
