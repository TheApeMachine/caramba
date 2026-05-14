//go:build !amd64 && !arm64

package activation

import "fmt"

func leakyReLUKernel(dst, src []float64, alpha float64) {
	if len(dst) != len(src) {
		panic(fmt.Sprintf("leakyReLUKernel: dst and src length mismatch: dst=%d src=%d", len(dst), len(src)))
	}

	for index, value := range src {
		if value < 0 {
			dst[index] = alpha * value

			continue
		}

		dst[index] = value
	}
}
