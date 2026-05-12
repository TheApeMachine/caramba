package math

// LogSumExp computes log(sum(exp(x))) over the last dimension via a dedicated
// AVX2/SSE2/NEON kernel — max, exp, sum, log all fused inline.
type LogSumExp struct{}

func NewLogSumExp() *LogSumExp { return &LogSumExp{} }

func (op *LogSumExp) Forward(shape []int, data ...[]float64) []float64 {
	x := data[0]
	dimSize := shape[len(shape)-1]
	n := len(x) / dimSize
	out := make([]float64, n)

	for i := 0; i < n; i++ {
		row := x[i*dimSize : (i+1)*dimSize]
		out[i] = logSumExpRowSIMD(row)
	}

	return out
}
