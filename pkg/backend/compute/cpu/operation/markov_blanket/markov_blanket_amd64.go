//go:build amd64

package markov_blanket

import (
	"fmt"

	"golang.org/x/sys/cpu"
)

var (
	useAVX2 bool
	useFMA  bool
)

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func matvecAVX2(dst, w, x []float64, rows, cols int)

//go:noescape
func matvecSSE2(dst, w, x []float64, rows, cols int)

//go:noescape
func subVecAVX2(dst, a, b []float64)

//go:noescape
func subVecSSE2(dst, a, b []float64)

func applyFlowInternal(out, xSens, wInt, bias []float64, Ni, Ns int) {
	if len(out) < Ni || len(bias) < Ni || len(xSens) < Ns || len(wInt) < Ni*Ns {
		panic(fmt.Errorf(
			"markov_blanket: applyFlowInternal: need len(out)>=%d, len(bias)>=%d, len(xSens)>=%d, len(wInt)>=%d",
			Ni, Ni, Ns, Ni*Ns,
		))
	}

	copy(out, bias)

	if useAVX2 && useFMA {
		matvecAVX2(out, wInt, xSens, Ni, Ns)
	} else {
		matvecSSE2(out, wInt, xSens, Ni, Ns)
	}
}

func applyFlowActive(out, xInt, wAct, bias []float64, Na, Ni int) {
	if len(out) < Na || len(bias) < Na || len(xInt) < Ni || len(wAct) < Na*Ni {
		panic(fmt.Errorf(
			"markov_blanket: applyFlowActive: need len(out)>=%d, len(bias)>=%d, len(xInt)>=%d, len(wAct)>=%d",
			Na, Na, Ni, Na*Ni,
		))
	}

	copy(out, bias)

	if useAVX2 && useFMA {
		matvecAVX2(out, wAct, xInt, Na, Ni)
	} else {
		matvecSSE2(out, wAct, xInt, Na, Ni)
	}
}

func alignedLen(n int) int {
	width := 2

	if useAVX2 {
		width = 4
	}

	return n / width * width
}
