//go:build amd64

package activation

import "golang.org/x/sys/cpu"

var useAVX2 bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
}

//go:noescape
func GeLUAVX2(dst, x []float64)

//go:noescape
func GeLUSSE2(dst, x []float64)

func applyGeLU(dst, src []float64) {
	width := 2

	if useAVX2 {
		width = 4
		limit := len(src) / width * width

		if limit > 0 {
			GeLUAVX2(dst[:limit], src[:limit])
		}

		scalarGeLU(dst[limit:], src[limit:])
		return
	}

	limit := len(src) / width * width

	if limit > 0 {
		GeLUSSE2(dst[:limit], src[:limit])
	}

	scalarGeLU(dst[limit:], src[limit:])
}
