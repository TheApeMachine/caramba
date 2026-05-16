//go:build darwin && cgo

package metal

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestConvolutionOps_Conv1dTensor(test *testing.T) {
	lib := metallibPathOrSkip(test, "convolution.metallib")

	Convey("Given resident Metal 1-D convolution tensors", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		convolutionOps, err := NewConvolutionOps(lib)
		So(err, ShouldBeNil)

		inputShape := mustMetalConvolutionShape(test, []int{1, 1, 4})
		weightShape := mustMetalConvolutionShape(test, []int{1, 1, 1})
		biasShape := mustMetalConvolutionShape(test, []int{1})
		outputShape := mustMetalConvolutionShape(test, []int{1, 1, 4})
		input := uploadMetalTensorForTest(
			test,
			tensorBackend,
			inputShape,
			[]float64{1, 2, 3, 4},
		)
		weight := uploadMetalTensorForTest(test, tensorBackend, weightShape, []float64{2})
		bias := uploadMetalTensorForTest(test, tensorBackend, biasShape, []float64{0.5})

		Convey("It should execute conv1d without host-staged dispatch", func() {
			output, err := convolutionOps.Conv1dTensor(
				input,
				weight,
				bias,
				outputShape,
				1,
				1,
				4,
				1,
				1,
				1,
				0,
				1,
				1,
			)
			So(err, ShouldBeNil)
			defer func() { So(output.Close(), ShouldBeNil) }()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			assertMetalConvolutionSlice(
				"conv1d_tensor",
				values,
				[]float64{2.5, 4.5, 6.5, 8.5},
				1e-6,
			)
		})
	})
}

func TestConvolutionOps_Conv2dTensor(test *testing.T) {
	lib := metallibPathOrSkip(test, "convolution.metallib")

	Convey("Given resident Metal 2-D convolution tensors", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		convolutionOps, err := NewConvolutionOps(lib)
		So(err, ShouldBeNil)

		inputShape := mustMetalConvolutionShape(test, []int{1, 1, 2, 2})
		weightShape := mustMetalConvolutionShape(test, []int{1, 1, 1, 1})
		biasShape := mustMetalConvolutionShape(test, []int{1})
		outputShape := mustMetalConvolutionShape(test, []int{1, 1, 2, 2})
		input := uploadMetalTensorForTest(
			test,
			tensorBackend,
			inputShape,
			[]float64{1, 2, 3, 4},
		)
		weight := uploadMetalTensorForTest(test, tensorBackend, weightShape, []float64{2})
		bias := uploadMetalTensorForTest(test, tensorBackend, biasShape, []float64{0.5})

		Convey("It should execute conv2d without host-staged dispatch", func() {
			output, err := convolutionOps.Conv2dTensor(
				input,
				weight,
				bias,
				outputShape,
				1,
				1,
				2,
				2,
				1,
				1,
				1,
				1,
				1,
				0,
				0,
				1,
				1,
				1,
			)
			So(err, ShouldBeNil)
			defer func() { So(output.Close(), ShouldBeNil) }()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			assertMetalConvolutionSlice(
				"conv2d_tensor",
				values,
				[]float64{2.5, 4.5, 6.5, 8.5},
				1e-6,
			)
		})
	})
}

func TestConvolutionOps_Conv3dTensor(test *testing.T) {
	lib := metallibPathOrSkip(test, "convolution.metallib")

	Convey("Given resident Metal 3-D convolution tensors", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		convolutionOps, err := NewConvolutionOps(lib)
		So(err, ShouldBeNil)

		inputShape := mustMetalConvolutionShape(test, []int{1, 1, 1, 2, 2})
		weightShape := mustMetalConvolutionShape(test, []int{1, 1, 1, 1, 1})
		biasShape := mustMetalConvolutionShape(test, []int{1})
		outputShape := mustMetalConvolutionShape(test, []int{1, 1, 1, 2, 2})
		input := uploadMetalTensorForTest(
			test,
			tensorBackend,
			inputShape,
			[]float64{1, 2, 3, 4},
		)
		weight := uploadMetalTensorForTest(test, tensorBackend, weightShape, []float64{2})
		bias := uploadMetalTensorForTest(test, tensorBackend, biasShape, []float64{0.5})

		Convey("It should execute conv3d without host-staged dispatch", func() {
			output, err := convolutionOps.Conv3dTensor(
				input,
				weight,
				bias,
				outputShape,
				1,
				1,
				1,
				2,
				2,
				1,
				1,
				1,
				1,
				1,
				1,
				1,
				0,
				0,
				0,
				1,
				1,
				1,
				1,
			)
			So(err, ShouldBeNil)
			defer func() { So(output.Close(), ShouldBeNil) }()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			assertMetalConvolutionSlice(
				"conv3d_tensor",
				values,
				[]float64{2.5, 4.5, 6.5, 8.5},
				1e-6,
			)
		})
	})
}

func TestConvolutionOps_ConvTranspose2dTensor(test *testing.T) {
	lib := metallibPathOrSkip(test, "convolution.metallib")

	Convey("Given resident Metal transposed convolution tensors", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		convolutionOps, err := NewConvolutionOps(lib)
		So(err, ShouldBeNil)

		inputShape := mustMetalConvolutionShape(test, []int{1, 1, 1, 2})
		weightShape := mustMetalConvolutionShape(test, []int{1, 1, 1, 1})
		biasShape := mustMetalConvolutionShape(test, []int{1})
		outputShape := mustMetalConvolutionShape(test, []int{1, 1, 1, 3})
		input := uploadMetalTensorForTest(test, tensorBackend, inputShape, []float64{1, 2})
		weight := uploadMetalTensorForTest(test, tensorBackend, weightShape, []float64{1})
		bias := uploadMetalTensorForTest(test, tensorBackend, biasShape, []float64{0.25})

		Convey("It should execute transposed conv2d without host-staged dispatch", func() {
			output, err := convolutionOps.ConvTranspose2dTensor(
				input,
				weight,
				bias,
				outputShape,
				1,
				1,
				1,
				2,
				1,
				1,
				1,
				1,
				2,
				0,
				0,
				1,
				1,
				1,
				0,
				0,
			)
			So(err, ShouldBeNil)
			defer func() { So(output.Close(), ShouldBeNil) }()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			assertMetalConvolutionSlice(
				"conv_transpose2d_tensor",
				values,
				[]float64{1.25, 0.25, 2.25},
				1e-6,
			)
		})
	})
}
