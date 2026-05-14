//go:build darwin && cgo

package metal

import (
	"math"
	"os"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestMetalTrainingOps(t *testing.T) {
	lib := metallibPathOrSkip(t, "math.metallib")

	Convey("Given initialized MetalTrainingOps", t, func() {
		trainingOps, err := NewTrainingOps(lib)
		So(err, ShouldBeNil)
		defer func() {
			So(trainingOps.Close(), ShouldBeNil)
		}()

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

		Convey("It should validate edge-case inputs", func() {
			_, err := trainingOps.MSELoss([]float64{1}, []float64{})
			So(err, ShouldNotBeNil)

			emptyLoss, err := trainingOps.MSELoss(nil, nil)
			So(err, ShouldBeNil)
			So(emptyLoss, ShouldResemble, []float64{0})

			_, err = trainingOps.MSEGrad([]float64{1}, []float64{})
			So(err, ShouldNotBeNil)

			emptyGrad, err := trainingOps.MSEGrad(nil, nil)
			So(err, ShouldBeNil)
			So(emptyGrad, ShouldBeEmpty)

			_, err = trainingOps.CrossEntropyLoss(nil, nil)
			So(err, ShouldNotBeNil)

			_, err = trainingOps.CrossEntropyLoss([]float64{1}, []float64{})
			So(err, ShouldNotBeNil)

			_, err = trainingOps.CrossEntropyGrad([]float64{1}, []float64{})
			So(err, ShouldNotBeNil)

			emptyCEGrad, err := trainingOps.CrossEntropyGrad(nil, nil)
			So(err, ShouldBeNil)
			So(emptyCEGrad, ShouldBeEmpty)

			_, err = trainingOps.Accuracy(nil, nil)
			So(err, ShouldNotBeNil)

			_, err = trainingOps.Accuracy([]float64{1}, []float64{})
			So(err, ShouldNotBeNil)

			emptyCounts, err := trainingOps.F1Counts(nil, nil)
			So(err, ShouldBeNil)
			So(emptyCounts, ShouldResemble, []float64{0, 0, 0})

			_, err = trainingOps.F1Counts([]float64{1}, []float64{})
			So(err, ShouldNotBeNil)
		})

		Convey("It should keep special floating point values explicit", func() {
			loss, err := trainingOps.MSELoss(
				[]float64{math.NaN(), math.Inf(1), math.Inf(-1)},
				[]float64{0, 0, 0},
			)

			So(err, ShouldBeNil)
			So(math.IsNaN(loss[0]) || math.IsInf(loss[0], 0), ShouldBeTrue)
		})

		Convey("It should handle large inputs without panicking", func() {
			largePredictions, largeTargets := metalTrainingBenchmarkData(4096)
			loss, err := trainingOps.MSELoss(largePredictions, largeTargets)

			So(err, ShouldBeNil)
			So(loss, ShouldHaveLength, 1)
		})
	})
}

func metalTrainingBenchmarkLib(b *testing.B, name string) string {
	b.Helper()

	lib := testdataPathMetalLib(name)
	if _, err := os.Stat(lib); err != nil {
		b.Skipf("missing %s; run `make build` in repo root", lib)
	}

	return lib
}

func metalTrainingBenchmarkData(size int) ([]float64, []float64) {
	predictions := make([]float64, size)
	targets := make([]float64, size)

	for index := range predictions {
		predictions[index] = float64(index%1024) / 1024
		targets[index] = float64((index+17)%1024) / 1024
	}

	return predictions, targets
}

func BenchmarkMetalTrainingOpsMSELoss(b *testing.B) {
	trainingOps, err := NewTrainingOps(metalTrainingBenchmarkLib(b, "math.metallib"))

	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		_ = trainingOps.Close()
	}()

	predictions, targets := metalTrainingBenchmarkData(10_000)

	for b.Loop() {
		if _, err := trainingOps.MSELoss(predictions, targets); err != nil {
			b.Fatalf("MSELoss failed: %v", err)
		}
	}
}

func BenchmarkMetalTrainingOpsMSEGrad(b *testing.B) {
	trainingOps, err := NewTrainingOps(metalTrainingBenchmarkLib(b, "math.metallib"))

	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		_ = trainingOps.Close()
	}()

	predictions, targets := metalTrainingBenchmarkData(10_000)

	for b.Loop() {
		if _, err := trainingOps.MSEGrad(predictions, targets); err != nil {
			b.Fatalf("MSEGrad failed: %v", err)
		}
	}
}

func BenchmarkMetalTrainingOpsCrossEntropyLoss(b *testing.B) {
	trainingOps, err := NewTrainingOps(metalTrainingBenchmarkLib(b, "math.metallib"))

	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		_ = trainingOps.Close()
	}()

	logits, targets := metalTrainingBenchmarkData(10_000)

	for b.Loop() {
		if _, err := trainingOps.CrossEntropyLoss(logits, targets); err != nil {
			b.Fatalf("CrossEntropyLoss failed: %v", err)
		}
	}
}

func BenchmarkMetalTrainingOpsCrossEntropyGrad(b *testing.B) {
	trainingOps, err := NewTrainingOps(metalTrainingBenchmarkLib(b, "math.metallib"))

	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		_ = trainingOps.Close()
	}()

	logits, targets := metalTrainingBenchmarkData(10_000)

	for b.Loop() {
		if _, err := trainingOps.CrossEntropyGrad(logits, targets); err != nil {
			b.Fatalf("CrossEntropyGrad failed: %v", err)
		}
	}
}

func BenchmarkMetalTrainingOpsAccuracy(b *testing.B) {
	trainingOps, err := NewTrainingOps(metalTrainingBenchmarkLib(b, "math.metallib"))

	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		_ = trainingOps.Close()
	}()

	predictions, targets := metalTrainingBenchmarkData(10_000)

	for b.Loop() {
		if _, err := trainingOps.Accuracy(predictions, targets); err != nil {
			b.Fatalf("Accuracy failed: %v", err)
		}
	}
}

func BenchmarkMetalTrainingOpsF1Counts(b *testing.B) {
	trainingOps, err := NewTrainingOps(metalTrainingBenchmarkLib(b, "math.metallib"))

	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		_ = trainingOps.Close()
	}()

	predictions, targets := metalTrainingBenchmarkData(10_000)

	for b.Loop() {
		if _, err := trainingOps.F1Counts(predictions, targets); err != nil {
			b.Fatalf("F1Counts failed: %v", err)
		}
	}
}
