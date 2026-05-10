//go:build amd64

package pooling

import "golang.org/x/sys/cpu"

var useAVX2 bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
}

//go:noescape
func reduceMaxAVX2(a []float64) float64

//go:noescape
func reduceMaxSSE2(a []float64) float64

//go:noescape
func reduceSumAVX2(a []float64) float64

//go:noescape
func reduceSumSSE2(a []float64) float64

func kernelMax(a []float64) float64 {
	if useAVX2 {
		return reduceMaxAVX2(a)
	}
	return reduceMaxSSE2(a)
}

func kernelSum(a []float64) float64 {
	if useAVX2 {
		return reduceSumAVX2(a)
	}
	return reduceSumSSE2(a)
}
