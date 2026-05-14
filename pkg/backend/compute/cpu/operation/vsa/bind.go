package vsa

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Bind computes the VSA binding operation (elementwise multiplication / Hadamard product).
In FHRR-style VSA, binding combines two hypervectors to represent a relationship;
the result is approximately orthogonal to both inputs when vectors are random.
shape=[N], data[0]=a, data[1]=b → out[N].
*/
type Bind struct{}

/*
NewBind instantiates a new Bind operation.
*/
func NewBind() *Bind { return &Bind{} }

/*
Forward computes elementwise product of data[0] and data[1].
*/
func (bind *Bind) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("vsa.bind", 2); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 1 {
		return nil, fmt.Errorf("vsa.bind: shape is required")
	}

	n := shape[0]

	if n < 0 {
		return nil, fmt.Errorf("vsa.bind: n must be non-negative, got %d", n)
	}

	if len(stateDict.Inputs[0]) != n || len(stateDict.Inputs[1]) != n {
		return nil, fmt.Errorf(
			"vsa.bind: inputs must both have length %d, got %d and %d",
			n, len(stateDict.Inputs[0]), len(stateDict.Inputs[1]),
		)
	}

	stateDict.EnsureOperationOutLen(n)
	bindKernel(stateDict.Out, stateDict.Inputs[0], stateDict.Inputs[1])

	return stateDict, nil
}
