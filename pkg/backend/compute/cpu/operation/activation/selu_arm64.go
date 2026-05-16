//go:build arm64

package activation

import (
	"fmt"

	computemath "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

//go:noescape
func seluBlendNEON(dst, src, expValues []float64)

func seluKernel(dst, src []float64) {
	if len(dst) < len(src) {
		panic(fmt.Sprintf("SELU: dst shorter than src (dst=%d src=%d)", len(dst), len(src)))
	}

	expValues := make([]float64, len(src))
	computemath.ExpVec(expValues, src)

	limit := len(src) / 2 * 2

	if limit > 0 {
		seluBlendNEON(dst[:limit], src[:limit], expValues[:limit])
	}

	seluBlendGeneric(dst[limit:len(src)], src[limit:len(src)], expValues[limit:len(src)])
}
