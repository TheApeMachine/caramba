//go:build amd64

package convolution

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA  bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA  = cpu.X86.HasFMA
}

//go:noescape
func dotProductAVX2(a, b []float64) float64

//go:noescape
func dotProductSSE2(a, b []float64) float64

//go:noescape
func scaledAddAVX2(dst, src []float64, scale float64)

//go:noescape
func scaledAddSSE2(dst, src []float64, scale float64)

func dotProduct(a, b []float64) float64 {
	if useAVX2 && useFMA {
		return dotProductAVX2(a, b)
	}
	return dotProductSSE2(a, b)
}

func scaledAdd(dst, src []float64, scale float64) {
	if useAVX2 {
		scaledAddAVX2(dst, src, scale)
	} else {
		scaledAddSSE2(dst, src, scale)
	}
}
