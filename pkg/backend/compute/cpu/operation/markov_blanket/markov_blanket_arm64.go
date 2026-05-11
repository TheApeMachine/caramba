//go:build arm64

package markov_blanket

//go:noescape
func matvecNEON(dst, w, x []float64, rows, cols int)

func applyFlowInternal(out, xSens, wInt, bias []float64, Ni, Ns int) {
	copy(out, bias)
	matvecNEON(out, wInt, xSens, Ni, Ns)
}

func applyFlowActive(out, xInt, wAct, bias []float64, Na, Ni int) {
	copy(out, bias)
	matvecNEON(out, wAct, xInt, Na, Ni)
}

func applyPartition(
	out, x, smask, amask, imask, emask []float64,
	Ns, Na, Ni, Ne int,
) {
	applyPartitionScalar(out, x, smask, amask, imask, emask, Ns, Na, Ni, Ne)
}
