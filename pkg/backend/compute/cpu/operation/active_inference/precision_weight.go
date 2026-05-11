package active_inference

import "fmt"

/*
PrecisionWeight scales prediction errors by precision (inverse variance),
implementing the precision-weighted prediction error central to Active Inference.

  out[i] = error[i] * exp(log_precision[i])

shape = [N], data[0] = error [N], data[1] = log_precision [N] → weighted error [N].
*/
type PrecisionWeight struct{}

/*
NewPrecisionWeight instantiates a new PrecisionWeight operation.
*/
func NewPrecisionWeight() *PrecisionWeight { return &PrecisionWeight{} }

/*
Forward computes precision-weighted prediction errors: out[i] = error[i] * exp(log_prec[i]).
*/
func (op *PrecisionWeight) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 1 {
		panic(fmt.Sprintf("active_inference: PrecisionWeight.Forward: len(shape)=%d, need >= 1", len(shape)))
	}

	n := shape[0]

	if len(data) < 2 {
		panic(fmt.Sprintf("active_inference: PrecisionWeight.Forward: len(data)=%d, need >= 2", len(data)))
	}

	if len(data[0]) != n {
		panic(fmt.Sprintf("active_inference: PrecisionWeight.Forward: len(error)=%d, need N=%d", len(data[0]), n))
	}

	if len(data[1]) != n {
		panic(fmt.Sprintf("active_inference: PrecisionWeight.Forward: len(log_precision)=%d, need N=%d", len(data[1]), n))
	}

	out := make([]float64, n)
	applyPrecisionWeight(out, data[0], data[1], n)

	return out
}
