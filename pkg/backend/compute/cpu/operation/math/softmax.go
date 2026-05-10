package math

import gomath "math"

// Softmax computes softmax over the last dimension of the input.
// data[0]=x, shape=[..., dim_size]; output has the same shape.
type Softmax struct{}

func NewSoftmax() *Softmax { return &Softmax{} }

func (op *Softmax) Forward(shape []int, data ...[]float64) []float64 {
	x := data[0]
	dimSize := shape[len(shape)-1]
	out := make([]float64, len(x))
	copy(out, x)
	n := len(x) / dimSize
	for i := 0; i < n; i++ {
		row := out[i*dimSize : (i+1)*dimSize]
		softmaxRow(row)
	}
	return out
}

// softmaxRow computes in-place softmax over a single row.
func softmaxRow(row []float64) {
	mx := reduceMax(row)
	for i, v := range row {
		row[i] = gomath.Exp(v - mx)
	}
	sum := reduceSum(row)
	divScalar(row, sum)
}
