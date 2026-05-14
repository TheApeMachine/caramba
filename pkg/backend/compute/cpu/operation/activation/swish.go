package activation

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

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
	if err := stateDict.RequireOperation("activation.swish"); err != nil {
		return nil, err
	}

	swishKernel(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}
