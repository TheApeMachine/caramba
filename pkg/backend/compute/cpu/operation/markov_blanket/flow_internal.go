package markov_blanket

import "fmt"

/*
FlowInternal computes the internal state update conditioned on sensory input:

	x_int_new = W_int @ x_sens + bias

shape = [N_i, N_s, N_i]  (reuse: out-dim, in-dim, out-dim for shape validation)
data[0] = x_sens   [N_s]
data[1] = W_int    [N_i * N_s]  (row-major)
data[2] = bias     [N_i]

Returns x_int_new [N_i].
*/
type FlowInternal struct{}

func NewFlowInternal() *FlowInternal { return &FlowInternal{} }

func (op *FlowInternal) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Errorf("markov_blanket: FlowInternal: len(shape)=%d, need >= 2", len(shape)).Error())
	}

	if len(data) < 3 {
		panic(fmt.Errorf("markov_blanket: FlowInternal: len(data)=%d, need 3", len(data)).Error())
	}

	if len(shape) >= 3 && shape[2] != shape[0] {
		panic(fmt.Errorf(
			"markov_blanket: FlowInternal: shape[2]=%d must match N_i=shape[0]=%d",
			shape[2], shape[0],
		).Error())
	}

	Ni, Ns := shape[0], shape[1]
	xSens := data[0]
	wInt := data[1]
	bias := data[2]

	if len(xSens) != Ns {
		panic(fmt.Errorf(
			"markov_blanket: FlowInternal: len(x_sens)=%d, need N_s=%d",
			len(xSens), Ns,
		).Error())
	}

	if len(wInt) != Ni*Ns {
		panic(fmt.Errorf(
			"markov_blanket: FlowInternal: len(W_int)=%d, need N_i*N_s=%d",
			len(wInt), Ni*Ns,
		).Error())
	}

	if len(bias) != Ni {
		panic(fmt.Errorf(
			"markov_blanket: FlowInternal: len(bias)=%d, need N_i=%d",
			len(bias), Ni,
		).Error())
	}

	out := make([]float64, Ni)
	applyFlowInternal(out, xSens, wInt, bias, Ni, Ns)

	return out
}

func applyFlowInternalScalar(out, xSens, wInt, bias []float64, Ni, Ns int) {
	for row := range Ni {
		acc := bias[row]

		for col := range Ns {
			acc += wInt[row*Ns+col] * xSens[col]
		}

		out[row] = acc
	}
}
