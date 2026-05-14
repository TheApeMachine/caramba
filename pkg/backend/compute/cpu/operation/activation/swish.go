package activation

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Swish applies x * sigmoid(x) elementwise using the backend activation kernels.
*/
type Swish struct{}

/*
NewSwish instantiates a stateless Swish operation.
*/
func NewSwish() *Swish {
	return &Swish{}
}

func (swish *Swish) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if stateDict == nil {
		return nil, fmt.Errorf("activation.swish: state dict is nil")
	}

	if err := stateDict.RequireOperation("activation.swish"); err != nil {
		return nil, err
	}

	if stateDict.Out == nil {
		return nil, fmt.Errorf("activation.swish: output tensor is nil")
	}

	SwishKernel(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}
