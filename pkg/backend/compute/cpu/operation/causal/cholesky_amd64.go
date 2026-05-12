//go:build amd64

package causal

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func choleskyRegAVX2(L []float64, n int, eps float64)

func choleskyReg(L []float64, n int, eps float64) {
	choleskyRegAVX2(L, n, eps)
}
