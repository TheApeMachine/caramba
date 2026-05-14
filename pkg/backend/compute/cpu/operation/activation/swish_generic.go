//go:build !amd64 && !arm64

package activation

import (
	"fmt"
	"math"
)

func SwishKernel(dst, src []float64) {
	if len(dst) != len(src) {
		panic(fmt.Sprintf("SwishKernel: dst and src length mismatch: dst=%d src=%d", len(dst), len(src)))
	}

	for index, value := range src {
		dst[index] = value / (1 + math.Exp(-value))
	}
}
