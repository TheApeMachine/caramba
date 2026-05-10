package math

import gomath "math"

// LogSumExp computes log(sum(exp(x))) over the last dimension, numerically
// stable via max subtraction.
// data[0]=x, shape=[..., dim_size].
// Output has shape=[...] (last dim reduced away).
type LogSumExp struct{}

func NewLogSumExp() *LogSumExp { return &LogSumExp{} }

func (op *LogSumExp) Forward(shape []int, data ...[]float64) []float64 {
	x := data[0]
	dimSize := shape[len(shape)-1]
	n := len(x) / dimSize
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		row := x[i*dimSize : (i+1)*dimSize]
		// find max
		mx := row[0]
		for _, v := range row[1:] {
			if v > mx {
				mx = v
			}
		}
		// sum exp(v - mx)
		sum := 0.0
		for _, v := range row {
			sum += gomath.Exp(v - mx)
		}
		out[i] = gomath.Log(sum) + mx
	}
	return out
}
