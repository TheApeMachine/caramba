package train_test

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/train"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func forwardTrain(
	operation interface {
		Forward(*state.Dict) (*state.Dict, error)
	},
	inputs ...[]float64,
) []float64 {
	stateDict := state.NewDict()
	stateDict.Inputs = append(stateDict.Inputs, inputs...)

	outputState, err := operation.Forward(stateDict)

	So(err, ShouldBeNil)

	return outputState.Out
}

func TestMSELoss(t *testing.T) {
	Convey("Given an MSELoss node", t, func() {
		op := train.NewMSELoss()

		Convey("It should compute mean squared error", func() {
			predictions := []float64{1.0, 2.0, 3.0}
			targets := []float64{1.0, 2.0, 3.0}
			result := forwardTrain(op, predictions, targets)

			So(result, ShouldHaveLength, 1)
			So(result[0], ShouldAlmostEqual, 0.0, 1e-9)
		})

		Convey("It should return nonzero loss for mismatched values", func() {
			result := forwardTrain(op, []float64{0, 0}, []float64{1, 1})

			So(result[0], ShouldAlmostEqual, 1.0, 1e-9)
		})
	})
}

func TestCrossEntropyLoss(t *testing.T) {
	Convey("Given a CrossEntropyLoss node", t, func() {
		op := train.NewCrossEntropyLoss()

		Convey("It should return near-zero loss for a confident correct prediction", func() {
			logits := []float64{10.0, 0.0, 0.0}
			targets := []float64{1.0, 0.0, 0.0}
			result := forwardTrain(op, logits, targets)

			So(result, ShouldHaveLength, 1)
			So(result[0], ShouldBeLessThan, 0.01)
		})

		Convey("It should return high loss for a confident wrong prediction", func() {
			logits := []float64{10.0, 0.0, 0.0}
			targets := []float64{0.0, 1.0, 0.0}
			result := forwardTrain(op, logits, targets)

			So(result[0], ShouldBeGreaterThan, 5.0)
		})
	})
}

func TestMSEGrad(t *testing.T) {
	Convey("Given an MSEGrad node", t, func() {
		op := train.NewMSEGrad()

		Convey("It should return zero gradients when predictions match targets", func() {
			xs := []float64{1.0, 2.0}
			result := forwardTrain(op, xs, xs)

			for _, g := range result {
				So(math.Abs(g), ShouldBeLessThan, 1e-9)
			}
		})

		Convey("It should return positive gradient when prediction exceeds target", func() {
			result := forwardTrain(op, []float64{2.0}, []float64{1.0})

			So(result[0], ShouldBeGreaterThan, 0)
		})
	})
}

func BenchmarkMSELoss(b *testing.B) {
	op := train.NewMSELoss()
	predictions := make([]float64, 1024)
	targets := make([]float64, 1024)

	b.ResetTimer()

	for b.Loop() {
		stateDict := state.NewDict()
		stateDict.Inputs = append(stateDict.Inputs, predictions, targets)
		_, _ = op.Forward(stateDict)
	}
}

func BenchmarkCrossEntropyLoss(b *testing.B) {
	op := train.NewCrossEntropyLoss()
	logits := make([]float64, 1024)
	targets := make([]float64, 1024)
	targets[0] = 1.0

	b.ResetTimer()

	for b.Loop() {
		stateDict := state.NewDict()
		stateDict.Inputs = append(stateDict.Inputs, logits, targets)
		_, _ = op.Forward(stateDict)
	}
}
