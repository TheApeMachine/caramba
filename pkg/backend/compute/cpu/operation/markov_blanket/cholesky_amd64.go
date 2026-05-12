//go:build amd64

package markov_blanket

import "golang.org/x/sys/cpu"

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func choleskyDecompAVX2(L []float64, n int) uint64

func choleskyDecomp(L []float64, n int) bool {
	return choleskyDecompAVX2(L, n) == 1
}
