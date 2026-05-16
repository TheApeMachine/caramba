//go:build amd64

package activation

import (
	"fmt"

	computemath "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

//go:noescape
func seluBlendAVX2(dst, src, expValues []float64)

//go:noescape
func seluBlendSSE2(dst, src, expValues []float64)

func seluKernel(dst, src []float64) {
	if len(dst) < len(src) {
		panic(fmt.Sprintf("SELU: dst shorter than src (dst=%d src=%d)", len(dst), len(src)))
	}

	expValues := make([]float64, len(src))
	computemath.ExpVec(expValues, src)
	seluBlend(dst, src, expValues)
}

func seluBlend(dst, src, expValues []float64) {
	elementCount := len(src)
	offset := 0

	if useAVX2 {
		limit := elementCount / 4 * 4

		if limit > 0 {
			seluBlendAVX2(dst[:limit], src[:limit], expValues[:limit])
			offset = limit
		}
	}

	limit := offset + (elementCount-offset)/2*2

	if limit > offset {
		seluBlendSSE2(dst[offset:limit], src[offset:limit], expValues[offset:limit])
		offset = limit
	}

	if offset < elementCount {
		seluBlendGeneric(dst[offset:elementCount], src[offset:elementCount], expValues[offset:elementCount])
	}
}
