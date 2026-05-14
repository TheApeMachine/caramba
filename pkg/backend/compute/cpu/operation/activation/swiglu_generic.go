//go:build !amd64 && !arm64

package activation

import (
	"fmt"
	"math"
)

func swigluKernel(dst, src []float64) {
	if len(src) != 2*len(dst) {
		panic(fmt.Sprintf(
			"swigluKernel: expected len(src)==2*len(dst), got len(dst)=%d len(src)=%d",
			len(dst), len(src),
		))
	}

	half := len(dst)

	for index := range dst {
		gate := src[index]
		value := src[half+index]
		dst[index] = gate / (1 + math.Exp(-gate)) * value
	}
}
