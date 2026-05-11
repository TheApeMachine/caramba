package train_test

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/train"
)

func TestMSELoss(t *testing.T) {
	Convey("Given an MSELoss node", t, func() {
		op := train.NewMSELoss()

		Convey("It should compute mean squared error", func() {
			predictions := []float64{1.0, 2.0, 3.0}
			targets := []float64{1.0, 2.0, 3.0}
			result := op.Forward(nil, predictions, targets)

			So(result, ShouldHaveLength, 1)
			So(result[0], ShouldAlmostEqual, 0.0, 1e-9)
		})

		Convey("It should return nonzero loss for mismatched values", func() {
			result := op.Forward(nil, []float64{0, 0}, []float64{1, 1})

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
			result := op.Forward(nil, logits, targets)

			So(result, ShouldHaveLength, 1)
			So(result[0], ShouldBeLessThan, 0.01)
		})

		Convey("It should return high loss for a confident wrong prediction", func() {
			logits := []float64{10.0, 0.0, 0.0}
			targets := []float64{0.0, 1.0, 0.0}
			result := op.Forward(nil, logits, targets)

			So(result[0], ShouldBeGreaterThan, 5.0)
		})
	})
}

func TestMSEGrad(t *testing.T) {
	Convey("Given an MSEGrad node", t, func() {
		op := train.NewMSEGrad()

		Convey("It should return zero gradients when predictions match targets", func() {
			xs := []float64{1.0, 2.0}
			result := op.Forward(nil, xs, xs)

			for _, g := range result {
				So(math.Abs(g), ShouldBeLessThan, 1e-9)
			}
		})

		Convey("It should return positive gradient when prediction exceeds target", func() {
			result := op.Forward(nil, []float64{2.0}, []float64{1.0})

			So(result[0], ShouldBeGreaterThan, 0)
		})
	})
}

func BenchmarkMSELoss(b *testing.B) {
	op := train.NewMSELoss()
	predictions := make([]float64, 1024)
	targets := make([]float64, 1024)

	b.ResetTimer()

	for range b.N {
		op.Forward(nil, predictions, targets)
	}
}

func BenchmarkCrossEntropyLoss(b *testing.B) {
	op := train.NewCrossEntropyLoss()
	logits := make([]float64, 1024)
	targets := make([]float64, 1024)
	targets[0] = 1.0

	b.ResetTimer()

	for range b.N {
		op.Forward(nil, logits, targets)
	}
}
