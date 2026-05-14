package markov_blanket

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

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

func (flowInternal *FlowInternal) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 2 {
		return nil, fmt.Errorf("markov_blanket.flow_internal: len(shape)=%d, need >= 2", len(shape))
	}

	if err := stateDict.RequireOperationInputs("markov_blanket.flow_internal", 3); err != nil {
		return nil, err
	}

	if len(shape) >= 3 && shape[2] != shape[0] {
		return nil, fmt.Errorf(
			"markov_blanket.flow_internal: shape[2]=%d must match N_i=shape[0]=%d",
			shape[2], shape[0],
		)
	}

	internalCount := shape[0]
	sensoryCount := shape[1]
	sensory := stateDict.Inputs[0]
	weight := stateDict.Inputs[1]
	bias := stateDict.Inputs[2]

	if len(sensory) != sensoryCount {
		return nil, fmt.Errorf(
			"markov_blanket.flow_internal: len(x_sens)=%d, need N_s=%d",
			len(sensory), sensoryCount,
		)
	}

	if len(weight) != internalCount*sensoryCount {
		return nil, fmt.Errorf(
			"markov_blanket.flow_internal: len(W_int)=%d, need N_i*N_s=%d",
			len(weight), internalCount*sensoryCount,
		)
	}

	if len(bias) != internalCount {
		return nil, fmt.Errorf(
			"markov_blanket.flow_internal: len(bias)=%d, need N_i=%d",
			len(bias), internalCount,
		)
	}

	stateDict.EnsureOperationOutLen(internalCount)
	applyFlowInternal(
		stateDict.Out,
		sensory,
		weight,
		bias,
		internalCount,
		sensoryCount,
	)

	return stateDict, nil
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
