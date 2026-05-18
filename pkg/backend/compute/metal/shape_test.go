//go:build darwin && cgo

package metal

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	cpushape "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/shape"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestMetalShapeOpsForward(t *testing.T) {
	Convey("Given MetalShapeOps Forward dispatch", t, func() {
		ops := &MetalShapeOps{}

		Convey("It should reject unsupported operation metadata", func() {
			output, err := ops.Forward(
				ShapeForwardRequest{Op: "shape.unknown"},
				[]int{2, 2},
				[]float64{1, 2, 3, 4},
			)

			So(err, ShouldNotBeNil)
			So(output, ShouldBeNil)
		})
	})
}

func TestMetalShapeOps_CopyTensor(test *testing.T) {
	lib := metallibPathOrSkip(test, "shape.metallib")

	Convey("Given a resident Metal tensor reshape", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		shapeOps, err := NewShapeOps(lib)
		So(err, ShouldBeNil)

		inputShape, err := computetensor.NewShape([]int{2, 3})
		So(err, ShouldBeNil)
		outputShape, err := computetensor.NewShape([]int{3, 2})
		So(err, ShouldBeNil)
		input := uploadMetalTensorForTest(
			test,
			tensorBackend,
			inputShape,
			[]float64{1, 2, 3, 4, 5, 6},
		)

		Convey("It should copy elements without leaving Metal storage", func() {
			output, err := shapeOps.CopyTensor(input, outputShape)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			So(output.Shape().Dims(), ShouldResemble, []int{3, 2})
			So(values, ShouldResemble, []float64{1, 2, 3, 4, 5, 6})
		})
	})
}

func TestMetalShapeOps_ReshapeTensor(test *testing.T) {
	lib := metallibPathOrSkip(test, "shape.metallib")

	Convey("Given a resident Metal tensor reshape", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		shapeOps, err := NewShapeOps(lib)
		So(err, ShouldBeNil)
		shapeOps.runtime = tensorBackend.runtime

		inputShape, err := computetensor.NewShape([]int{2, 3})
		So(err, ShouldBeNil)
		outputShape, err := computetensor.NewShape([]int{3, 2})
		So(err, ShouldBeNil)
		input := uploadMetalTensorForTest(
			test,
			tensorBackend,
			inputShape,
			[]float64{1, 2, 3, 4, 5, 6},
		)

		Convey("It should change metadata without allocating output storage", func() {
			before := tensorBackend.runtime.Metrics()
			output, err := shapeOps.ReshapeTensor(input, outputShape)
			after := tensorBackend.runtime.Metrics()
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			So(output.Shape().Dims(), ShouldResemble, []int{3, 2})
			So(values, ShouldResemble, []float64{1, 2, 3, 4, 5, 6})
			So(after.AllocatedBytes, ShouldEqual, before.AllocatedBytes)
		})
	})
}

func TestMetalShapeOps_TransposeTensor(test *testing.T) {
	lib := metallibPathOrSkip(test, "shape.metallib")

	Convey("Given a resident Metal tensor transpose", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		shapeOps, err := NewShapeOps(lib)
		So(err, ShouldBeNil)

		inputShape, err := computetensor.NewShape([]int{2, 3})
		So(err, ShouldBeNil)
		outputShape, err := computetensor.NewShape([]int{3, 2})
		So(err, ShouldBeNil)
		input := uploadMetalTensorForTest(
			test,
			tensorBackend,
			inputShape,
			[]float64{1, 2, 3, 4, 5, 6},
		)

		Convey("It should transpose dimensions without leaving Metal storage", func() {
			output, err := shapeOps.TransposeTensor(input, outputShape, 0, 1)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			So(output.Shape().Dims(), ShouldResemble, []int{3, 2})
			So(values, ShouldResemble, []float64{1, 4, 2, 5, 3, 6})
		})
	})
}

func BenchmarkMetalShapeOps_ReshapeTensor(benchmark *testing.B) {
	lib := metallibPathOrSkip(benchmark, "shape.metallib")
	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}

	defer func() {
		_ = tensorBackend.Close()
	}()

	shapeOps, err := NewShapeOps(lib)
	if err != nil {
		benchmark.Fatal(err)
	}

	shapeOps.runtime = tensorBackend.runtime
	inputShape, err := computetensor.NewShape([]int{64, 128})
	if err != nil {
		benchmark.Fatal(err)
	}

	outputShape, err := computetensor.NewShape([]int{128, 64})
	if err != nil {
		benchmark.Fatal(err)
	}

	input := uploadMetalTensor(tensorBackend, inputShape, make([]float64, inputShape.Len()))
	defer closeBenchmarkTensors(input)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := shapeOps.ReshapeTensor(input, outputShape)
		closeBenchmarkOutput(benchmark, output, err)
	}
}

func TestMetalShapeOps_ConcatTensor(test *testing.T) {
	lib := metallibPathOrSkip(test, "shape.metallib")

	Convey("Given resident Metal tensors to concatenate", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		shapeOps, err := NewShapeOps(lib)
		So(err, ShouldBeNil)

		leftShape, err := computetensor.NewShape([]int{3})
		So(err, ShouldBeNil)
		rightShape, err := computetensor.NewShape([]int{2})
		So(err, ShouldBeNil)
		outputShape, err := computetensor.NewShape([]int{5})
		So(err, ShouldBeNil)
		left := uploadMetalTensorForTest(test, tensorBackend, leftShape, []float64{1, 2, 3})
		right := uploadMetalTensorForTest(test, tensorBackend, rightShape, []float64{4, 5})

		Convey("It should concatenate without leaving Metal storage", func() {
			output, err := shapeOps.ConcatTensor(left, right, outputShape)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2, 3, 4, 5})
		})
	})
}

func TestMetalShapeOps_SplitTensor(test *testing.T) {
	lib := metallibPathOrSkip(test, "shape.metallib")

	Convey("Given a resident Metal tensor split", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		shapeOps, err := NewShapeOps(lib)
		So(err, ShouldBeNil)

		inputShape, err := computetensor.NewShape([]int{2, 4})
		So(err, ShouldBeNil)
		outputShape, err := computetensor.NewShape([]int{8})
		So(err, ShouldBeNil)
		input := uploadMetalTensorForTest(
			test,
			tensorBackend,
			inputShape,
			[]float64{1, 2, 3, 4, 5, 6, 7, 8},
		)

		Convey("It should split chunks without leaving Metal storage", func() {
			output, err := shapeOps.SplitTensor(input, outputShape, 2, 4, 2, 1)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2, 5, 6, 3, 4, 7, 8})
		})
	})
}

func TestMetalShapeOpsForwardParity(t *testing.T) {
	lib := metallibPathOrSkip(t, "shape.metallib")

	Convey("Given initialized MetalShapeOps Forward dispatch", t, func() {
		ops, err := NewShapeOps(lib)
		So(err, ShouldBeNil)

		Convey("It should route transpose metadata to the Metal transpose kernel", func() {
			input := []float64{1, 2, 3, 4, 5, 6}
			expectedState := state.NewDict().WithShape([]int{2, 3}).WithInput(input)
			expectedState.Dim0 = 0
			expectedState.Dim1 = 1
			var err error
			expectedState, err = cpushape.NewTranspose(0, 1).Forward(expectedState)

			So(err, ShouldBeNil)
			expected := expectedState.Out

			output, err := ops.Forward(
				ShapeForwardRequest{
					Op: "shape.transpose",
					Metadata: map[string]any{
						"dim0": 0,
						"dim1": 1,
					},
				},
				[]int{2, 3},
				input,
			)

			So(err, ShouldBeNil)
			So(output, ShouldResemble, expected)
		})

		Convey("It should route view_as_heads metadata to the Metal head-view kernel", func() {
			input := []float64{1, 2, 3, 4, 5, 6, 7, 8}
			expectedState := state.NewDict().
				WithShape([]int{1, 2, 4}).
				WithInput(input)
			expectedState.NumHeads = 2
			var err error
			expectedState, err = cpushape.NewViewAsHeads(2).Forward(expectedState)

			So(err, ShouldBeNil)
			expected := expectedState.Out

			output, err := ops.Forward(
				ShapeForwardRequest{
					Op: "shape.view_as_heads",
					Metadata: map[string]any{
						"num_heads": 2,
					},
				},
				[]int{1, 2, 4},
				input,
			)

			So(err, ShouldBeNil)
			So(output, ShouldResemble, expected)
		})

		Convey("It should route merge_heads metadata to the Metal merge-heads kernel", func() {
			input := []float64{1, 3, 5, 7, 2, 4, 6, 8}
			expectedState, err := cpushape.NewMergeHeads().Forward(
				state.NewDict().WithShape([]int{1, 2, 2, 2}).WithInput(input),
			)

			So(err, ShouldBeNil)
			expected := expectedState.Out

			output, err := ops.Forward(
				ShapeForwardRequest{Op: "shape.merge_heads"},
				[]int{1, 2, 2, 2},
				input,
			)

			So(err, ShouldBeNil)
			So(output, ShouldResemble, expected)
		})

		Convey("It should run split through the Metal split kernel", func() {
			input := []float64{1, 2, 3, 4, 5, 6, 7, 8}
			expectedState := state.NewDict().WithShape([]int{2, 4}).WithInput(input)
			var err error
			expectedState.Dim = 1
			expectedState.SplitSize = 2
			expectedState, err = cpushape.NewSplit().Forward(expectedState)

			So(err, ShouldBeNil)

			output, err := ops.Split(input, 2, 4, 2, 1)

			So(err, ShouldBeNil)
			So(output, ShouldResemble, expectedState.Out)
		})

		Convey("It should route upsample_nearest2d metadata to the Metal upsample kernel", func() {
			input := []float64{1, 2, 3, 4}
			expectedState := state.NewDict().WithShape([]int{1, 1, 2, 2}).WithInput(input)
			expectedState.ScaleH = 2
			expectedState.ScaleW = 2
			var err error
			expectedState, err = cpushape.NewUpsampleNearest2D().Forward(expectedState)

			So(err, ShouldBeNil)

			output, err := ops.Forward(
				ShapeForwardRequest{
					Op: "shape.upsample_nearest2d",
					Metadata: map[string]any{
						"scale_h": 2,
						"scale_w": 2,
					},
				},
				[]int{1, 1, 2, 2},
				input,
			)

			So(err, ShouldBeNil)
			So(output, ShouldResemble, expectedState.Out)
		})
	})
}

func BenchmarkMetalShapeOpsForwardValidation(b *testing.B) {
	ops := &MetalShapeOps{}
	input := []float64{1, 2, 3, 4}

	for b.Loop() {
		_, _ = ops.Forward(ShapeForwardRequest{Op: "shape.unknown"}, []int{2, 2}, input)
	}
}
