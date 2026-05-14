//go:build amd64

package activation

import (
	"fmt"

	"golang.org/x/sys/cpu"
)

var useAVX2 bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
}

//go:noescape
func GeLUAVX2(dst, x []float64)

//go:noescape
func GeLUSSE2(dst, x []float64)

func geluKernel(dst, src []float64) {
	if len(dst) != len(src) {
		panic(fmt.Sprintf("geluKernel: dst and src length mismatch: dst=%d src=%d", len(dst), len(src)))
	}

	width := 2
	vectorGeLU := GeLUSSE2

	if useAVX2 {
		width = 4
		vectorGeLU = GeLUAVX2
	}

	limit := len(src) / width * width

	if limit > 0 {
		vectorGeLU(dst[:limit], src[:limit])
	}

	scalarGeLU(dst[limit:], src[limit:])
}
