package markov_blanket

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

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

func (flowActive *FlowActive) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 2 {
		return nil, fmt.Errorf("markov_blanket.flow_active: len(shape)=%d, need >= 2", len(shape))
	}

	if err := stateDict.RequireOperationInputs("markov_blanket.flow_active", 3); err != nil {
		return nil, err
	}

	activeCount := shape[0]
	internalCount := shape[1]
	internal := stateDict.Inputs[0]
	weight := stateDict.Inputs[1]
	bias := stateDict.Inputs[2]

	if len(internal) != internalCount {
		return nil, fmt.Errorf(
			"markov_blanket.flow_active: len(x_int)=%d, need N_i=%d",
			len(internal), internalCount,
		)
	}

	if len(weight) != activeCount*internalCount {
		return nil, fmt.Errorf(
			"markov_blanket.flow_active: len(W_act)=%d, need N_a*N_i=%d",
			len(weight), activeCount*internalCount,
		)
	}

	if len(bias) != activeCount {
		return nil, fmt.Errorf(
			"markov_blanket.flow_active: len(bias)=%d, need N_a=%d",
			len(bias), activeCount,
		)
	}

	stateDict.EnsureOperationOutLen(activeCount)
	applyFlowActive(
		stateDict.Out,
		internal,
		weight,
		bias,
		activeCount,
		internalCount,
	)

	return stateDict, nil
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
