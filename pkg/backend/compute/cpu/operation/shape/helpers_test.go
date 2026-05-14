package shape

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/state"

	. "github.com/smartystreets/goconvey/convey"
)

func forwardShape(
	operation interface {
		Forward(*state.Dict) (*state.Dict, error)
	},
	shape []int,
	input []float64,
	configure ...func(*state.Dict),
) []float64 {
	stateDict := state.NewDict().
		WithShape(shape).
		WithInput(input)

	for _, configureState := range configure {
		configureState(stateDict)
	}

	outputState, err := operation.Forward(stateDict)

	So(err, ShouldBeNil)

	return outputState.Out
}
