package operation

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Operation applies one backend-native operation to a state dictionary.
*/
type Operation interface {
	Forward(*state.Dict) (*state.Dict, error)
}

/*
Parameterized is implemented by operations that own learnable parameters.
Weights/LoadWeights on the Graph use this to snapshot and restore model state.
*/
type Parameterized interface {
	Params() []float64
	SetParams([]float64)
}
