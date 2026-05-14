//go:build !amd64 && !arm64

package activation

import (
	"fmt"
	"math"
)

func tanhKernel(dst, src []float64) {
	if len(dst) != len(src) {
		panic(fmt.Sprintf("tanhKernel: dst and src length mismatch: dst=%d src=%d", len(dst), len(src)))
	}

	for index, value := range src {
		dst[index] = math.Tanh(value)
	}
}
