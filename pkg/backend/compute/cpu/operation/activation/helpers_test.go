package activation

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/state"

	. "github.com/smartystreets/goconvey/convey"
)

type stateOperation interface {
	Forward(*state.Dict) (*state.Dict, error)
}

func forwardActivation(
	operation stateOperation, input []float64, configure ...func(*state.Dict),
) []float64 {
	stateDict := state.NewDict().
		WithShape([]int{len(input)}).
		WithInput(input)

	for _, configureState := range configure {
		configureState(stateDict)
	}

	outputState, err := operation.Forward(stateDict)

	So(err, ShouldBeNil)

	return outputState.Out
}
