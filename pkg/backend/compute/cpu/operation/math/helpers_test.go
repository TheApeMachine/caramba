package math

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/state"

	. "github.com/smartystreets/goconvey/convey"
)

type stateOperation interface {
	Forward(*state.Dict) (*state.Dict, error)
}

func forwardMath(
	operation stateOperation, shape []int, inputs ...[]float64,
) []float64 {
	stateDict := state.NewDict().WithShape(shape)

	values := make([]any, len(inputs))
	for index := range inputs {
		values[index] = inputs[index]
	}

	stateDict.WithInputs(values...)
	outputState, err := operation.Forward(stateDict)

	So(err, ShouldBeNil)

	return outputState.Out
}
