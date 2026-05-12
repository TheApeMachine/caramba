package math

// LogSumExp computes log(sum(exp(x))) over the last dimension, numerically
// stable via max subtraction.  Uses SIMD primitives for max / exp / sum / log.
// data[0]=x, shape=[..., dim_size].
// Output has shape=[...] (last dim reduced away).
type LogSumExp struct{}

func NewLogSumExp() *LogSumExp { return &LogSumExp{} }

func (op *LogSumExp) Forward(shape []int, data ...[]float64) []float64 {
	x := data[0]
	dimSize := shape[len(shape)-1]
	n := len(x) / dimSize
	out := make([]float64, n)
	scratch := make([]float64, dimSize)
	one := make([]float64, 1)

	for i := 0; i < n; i++ {
		row := x[i*dimSize : (i+1)*dimSize]
		mx := reduceMax(row)
		copy(scratch, row)
		addScalarVec(scratch, -mx)
		expVec(scratch, scratch)
		sum := reduceSum(scratch)
		one[0] = sum
		logVec(one, one)
		out[i] = one[0] + mx
	}

	return out
}
