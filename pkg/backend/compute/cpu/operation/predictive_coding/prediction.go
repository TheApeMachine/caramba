package predictive_coding

import "fmt"

/*
Prediction computes the top-down prediction μ̂ = W @ r from a higher-level
representation r through generative weights W. This is the forward pass of the
generative model in a predictive coding hierarchy.
shape=[D_out, D_in], data[0]=W [D_out*D_in], data[1]=r [D_in] → μ̂ [D_out].
*/
type Prediction struct{}

/*
NewPrediction instantiates a new Prediction operation.
*/
func NewPrediction() *Prediction { return &Prediction{} }

/*
Forward computes μ̂ = W @ r where W is [D_out, D_in] row-major and r is [D_in].
*/
func (op *Prediction) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Sprintf("predictive_coding: Prediction.Forward: len(shape)=%d, need >= 2", len(shape)))
	}

	dOut, dIn := shape[0], shape[1]

	if len(data) < 2 {
		panic(fmt.Sprintf("predictive_coding: Prediction.Forward: len(data)=%d, need >= 2", len(data)))
	}

	if len(data[0]) != dOut*dIn {
		panic(fmt.Sprintf(
			"predictive_coding: Prediction.Forward: len(W)=%d, need D_out*D_in=%d",
			len(data[0]), dOut*dIn,
		))
	}

	if len(data[1]) != dIn {
		panic(fmt.Sprintf(
			"predictive_coding: Prediction.Forward: len(r)=%d, need D_in=%d",
			len(data[1]), dIn,
		))
	}

	out := make([]float64, dOut)
	applyMatVec(out, data[0], data[1], dOut, dIn)

	return out
}
