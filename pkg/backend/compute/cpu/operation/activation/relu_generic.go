//go:build !amd64 && !arm64

package activation

import "fmt"

func reluKernel(dst, src []float64) {
	if len(dst) != len(src) {
		panic(fmt.Sprintf("reluKernel: dst and src length mismatch: dst=%d src=%d", len(dst), len(src)))
	}

	for index, value := range src {
		if value > 0 {
			dst[index] = value
		}
	}
}
