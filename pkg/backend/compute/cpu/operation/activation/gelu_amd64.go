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
	if useAVX2 {
		GeLUAVX2(dst, src)
	} else {
		GeLUSSE2(dst, src)
	}
}
