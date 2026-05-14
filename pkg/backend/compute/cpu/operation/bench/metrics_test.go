package bench_test

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/bench"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func forwardBench(
	operation interface {
		Forward(*state.Dict) (*state.Dict, error)
	},
	stateDict *state.Dict,
	predicted, targets []float64,
) []float64 {
	stateDict.Inputs = append(stateDict.Inputs[:0], predicted, targets)

	outputState, err := operation.Forward(stateDict)

	So(err, ShouldBeNil)

	return outputState.Out
}

func TestAccuracy(t *testing.T) {
	Convey("Given an Accuracy metric node", t, func() {
		op := bench.NewAccuracy()
		stateDict := state.NewDict()

		Convey("It should return 1.0 when predicted class matches target", func() {
			predicted := []float64{0.1, 0.9, 0.0}
			targets := []float64{0.0, 1.0, 0.0}
			result := forwardBench(op, stateDict, predicted, targets)

			So(result, ShouldHaveLength, 1)
			So(result[0], ShouldAlmostEqual, 1.0, 1e-9)
		})

		Convey("It should return 0.5 after one correct and one wrong prediction", func() {
			stateDict := state.NewDict()
			forwardBench(op, stateDict, []float64{0.9, 0.1}, []float64{1.0, 0.0})
			result := forwardBench(op, stateDict, []float64{0.1, 0.9}, []float64{1.0, 0.0})

			So(result[0], ShouldAlmostEqual, 0.5, 1e-9)
		})
	})
}

func TestPerplexity(t *testing.T) {
	Convey("Given a Perplexity metric node", t, func() {
		op := bench.NewPerplexity()
		stateDict := state.NewDict()

		Convey("It should return 1.0 for a perfect prediction", func() {
			probs := []float64{1.0, 0.0}
			targets := []float64{1.0, 0.0}
			result := forwardBench(op, stateDict, probs, targets)

			So(result, ShouldHaveLength, 1)
			So(result[0], ShouldBeLessThan, 1.1)
		})
	})
}

func TestF1(t *testing.T) {
	Convey("Given an F1 metric node", t, func() {
		op := bench.NewF1()
		stateDict := state.NewDict()

		Convey("It should return 1.0 for perfect binary classification", func() {
			predicted := []float64{1.0, 0.0, 1.0}
			targets := []float64{1.0, 0.0, 1.0}
			result := forwardBench(op, stateDict, predicted, targets)

			So(result, ShouldHaveLength, 1)
			So(result[0], ShouldAlmostEqual, 1.0, 1e-3)
		})

		Convey("It should return less than 1.0 for imperfect classification", func() {
			predicted := []float64{1.0, 1.0, 0.0}
			targets := []float64{1.0, 0.0, 1.0}
			result := forwardBench(op, state.NewDict(), predicted, targets)

			So(result[0], ShouldBeLessThan, 1.0)
		})
	})
}

func BenchmarkAccuracy(b *testing.B) {
	op := bench.NewAccuracy()
	stateDict := state.NewDict()
	predicted := make([]float64, 1000)
	targets := make([]float64, 1000)
	predicted[42] = 1.0
	targets[42] = 1.0

	b.ResetTimer()

	for b.Loop() {
		stateDict.Inputs = append(stateDict.Inputs[:0], predicted, targets)
		_, _ = op.Forward(stateDict)
	}
}
