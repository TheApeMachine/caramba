//go:build arm64

package markov_blanket

import "fmt"

// matvecNEON accumulates W@x into dst (row-major W); assembly in markov_blanket_neon_arm64.s.
//
//go:noescape
func matvecNEON(dst, w, x []float64, rows, cols int)

func applyFlowInternal(out, xSens, wInt, bias []float64, Ni, Ns int) {
	if len(out) < Ni || len(bias) < Ni || len(xSens) < Ns || len(wInt) < Ni*Ns {
		panic(fmt.Errorf(
			"markov_blanket: applyFlowInternal: need len(out)>=%d, len(bias)>=%d, len(xSens)>=%d, len(wInt)>=%d",
			Ni, Ni, Ns, Ni*Ns,
		))
	}

	copy(out, bias)
	matvecNEON(out, wInt, xSens, Ni, Ns)
}

func applyFlowActive(out, xInt, wAct, bias []float64, Na, Ni int) {
	if len(out) < Na || len(bias) < Na || len(xInt) < Ni || len(wAct) < Na*Ni {
		panic(fmt.Errorf(
			"markov_blanket: applyFlowActive: need len(out)>=%d, len(bias)>=%d, len(xInt)>=%d, len(wAct)>=%d",
			Na, Na, Ni, Na*Ni,
		))
	}

	copy(out, bias)
	matvecNEON(out, wAct, xInt, Na, Ni)
}
