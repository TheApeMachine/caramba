package markov_blanket

import "fmt"

/*
FlowActive computes the active state update conditioned on internal state:

	x_act_new = W_act @ x_int + bias

shape = [N_a, N_i]
data[0] = x_int  [N_i]
data[1] = W_act  [N_a * N_i]  (row-major)
data[2] = bias   [N_a]

Returns x_act_new [N_a].
*/
type FlowActive struct{}

func NewFlowActive() *FlowActive { return &FlowActive{} }

func (op *FlowActive) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Errorf("markov_blanket: FlowActive: len(shape)=%d, need >= 2", len(shape)).Error())
	}

	if len(data) < 3 {
		panic(fmt.Errorf("markov_blanket: FlowActive: len(data)=%d, need 3", len(data)).Error())
	}

	Na, Ni := shape[0], shape[1]
	xInt := data[0]
	wAct := data[1]
	bias := data[2]

	if len(xInt) != Ni {
		panic(fmt.Errorf(
			"markov_blanket: FlowActive: len(x_int)=%d, need N_i=%d",
			len(xInt), Ni,
		).Error())
	}

	if len(wAct) != Na*Ni {
		panic(fmt.Errorf(
			"markov_blanket: FlowActive: len(W_act)=%d, need N_a*N_i=%d",
			len(wAct), Na*Ni,
		).Error())
	}

	if len(bias) != Na {
		panic(fmt.Errorf(
			"markov_blanket: FlowActive: len(bias)=%d, need N_a=%d",
			len(bias), Na,
		).Error())
	}

	out := make([]float64, Na)
	applyFlowActive(out, xInt, wAct, bias, Na, Ni)

	return out
}

func applyFlowActiveScalar(out, xInt, wAct, bias []float64, Na, Ni int) {
	for row := range Na {
		acc := bias[row]

		for col := range Ni {
			acc += wAct[row*Ni+col] * xInt[col]
		}

		out[row] = acc
	}
}
