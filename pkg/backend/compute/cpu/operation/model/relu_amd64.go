//go:build amd64

package model

import "golang.org/x/sys/cpu"

var useAVX2 bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
}

//go:noescape
func reluAVX2(dst []float64)

//go:noescape
func reluSSE2(dst []float64)

func reluInPlace(x []float64) {
	if useAVX2 {
		reluAVX2(x)
	} else {
		reluSSE2(x)
	}
}
