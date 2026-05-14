package math

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Log computes log(x) elementwise via vectorized SIMD assembly.
*/
type Log struct{}

func NewLog() *Log { return &Log{} }

func (log *Log) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("math.log"); err != nil {
		return nil, err
	}

	logKernel(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}
