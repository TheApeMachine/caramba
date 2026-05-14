//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestMetalTrainingOps(t *testing.T) {
	lib := metallibPathOrSkip(t, "math.metallib")

	Convey("Given initialized MetalTrainingOps", t, func() {
		trainingOps, err := NewTrainingOps(lib)
		So(err, ShouldBeNil)

		predictions := []float64{0.2, 0.8, 0.4}
		targets := []float64{0, 1, 1}

		Convey("It should compute MSE loss and gradient on Metal", func() {
			loss, err := trainingOps.MSELoss(predictions, targets)

			So(err, ShouldBeNil)
			So(loss[0], ShouldAlmostEqual, 0.1466666667, 1e-5)

			grad, err := trainingOps.MSEGrad(predictions, targets)

			So(err, ShouldBeNil)
			So(grad[0], ShouldAlmostEqual, 0.1333333333, 1e-5)
			So(grad[1], ShouldAlmostEqual, -0.1333333333, 1e-5)
			So(grad[2], ShouldAlmostEqual, -0.4, 1e-5)
		})

		Convey("It should compute cross entropy loss and gradient on Metal", func() {
			logits := []float64{1, 2, 3}
			oneHot := []float64{0, 0, 1}
			loss, err := trainingOps.CrossEntropyLoss(logits, oneHot)
			expectedLoss := -math.Log(math.Exp(3)/(math.Exp(1)+math.Exp(2)+math.Exp(3)) + 1e-9)

			So(err, ShouldBeNil)
			So(loss[0], ShouldAlmostEqual, expectedLoss, 1e-5)

			grad, err := trainingOps.CrossEntropyGrad(logits, oneHot)

			So(err, ShouldBeNil)
			So(grad[0], ShouldAlmostEqual, 0.09003057, 1e-5)
			So(grad[1], ShouldAlmostEqual, 0.24472847, 1e-5)
			So(grad[2], ShouldAlmostEqual, -0.33475904, 1e-5)
		})

		Convey("It should compute benchmark primitives on Metal", func() {
			accuracy, err := trainingOps.Accuracy([]float64{0.1, 0.9}, []float64{0, 1})

			So(err, ShouldBeNil)
			So(accuracy[0], ShouldEqual, 1)

			counts, err := trainingOps.F1Counts(
				[]float64{0.9, 0.8, 0.1, 0.7},
				[]float64{1, 0, 1, 0},
			)

			So(err, ShouldBeNil)
			So(counts, ShouldResemble, []float64{1, 2, 1})
		})
	})
}

func BenchmarkMetalTrainingOpsMSELoss(b *testing.B) {
	trainingOps, err := NewTrainingOps(testdataPathMetalLib("math.metallib"))

	if err != nil {
		b.Fatal(err)
	}

	predictions := []float64{0.2, 0.8, 0.4}
	targets := []float64{0, 1, 1}

	for b.Loop() {
		_, _ = trainingOps.MSELoss(predictions, targets)
	}
}
