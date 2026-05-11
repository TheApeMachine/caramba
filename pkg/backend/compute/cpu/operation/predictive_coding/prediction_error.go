package predictive_coding

import "fmt"

/*
PredictionError computes the precision-weighted prediction error ε = Π ⊙ (x - μ̂)
where Π is the precision (inverse variance) diagonal. When no precision is provided
(data has only 2 elements), plain error x - μ̂ is returned.
shape=[N], data[0]=x [N], data[1]=mu_hat [N], data[2]=precision [N] (optional) → ε [N].
*/
type PredictionError struct{}

/*
NewPredictionError instantiates a new PredictionError operation.
*/
func NewPredictionError() *PredictionError { return &PredictionError{} }

/*
Forward computes ε = Π ⊙ (x - μ̂).
*/
func (op *PredictionError) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 1 {
		panic(fmt.Sprintf("predictive_coding: PredictionError.Forward: empty shape"))
	}

	n := shape[0]

	if len(data) < 2 {
		panic(fmt.Sprintf("predictive_coding: PredictionError.Forward: len(data)=%d, need >= 2", len(data)))
	}

	if len(data[0]) != n || len(data[1]) != n {
		panic(fmt.Sprintf(
			"predictive_coding: PredictionError.Forward: shape mismatch x=%d mu_hat=%d n=%d",
			len(data[0]), len(data[1]), n,
		))
	}

	out := make([]float64, n)
	applySubVec(out, data[0], data[1])

	if len(data) >= 3 {
		if len(data[2]) != n {
			panic(fmt.Sprintf(
				"predictive_coding: PredictionError.Forward: len(precision)=%d, need n=%d",
				len(data[2]), n,
			))
		}

		applyMulVec(out, out, data[2])
	}

	return out
}
