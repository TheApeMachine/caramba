//go:build linux && cgo && cuda

package cuda

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestTensorBackend_CopyTensor(test *testing.T) {
	Convey("Given a resident CUDA tensor reshape", test, func() {
		tensorBackend := newCUDATensorBackendForTest(test)
		inputShape := mustCUDAShape(test, []int{2, 3})
		outputShape := mustCUDAShape(test, []int{3, 2})
		input := uploadCUDATensorForTest(
			test,
			tensorBackend,
			inputShape,
			[]float64{1, 2, 3, 4, 5, 6},
		)

		Convey("It should copy elements without staging through host dispatch", func() {
			output, err := tensorBackend.CopyTensor(input, outputShape)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(output.Shape().Dims(), ShouldResemble, []int{3, 2})
			So(values, ShouldResemble, []float64{1, 2, 3, 4, 5, 6})
		})
	})
}

func TestTensorBackend_TransposeTensor(test *testing.T) {
	Convey("Given a resident CUDA tensor transpose", test, func() {
		tensorBackend := newCUDATensorBackendForTest(test)
		inputShape := mustCUDAShape(test, []int{2, 3})
		outputShape := mustCUDAShape(test, []int{3, 2})
		input := uploadCUDATensorForTest(
			test,
			tensorBackend,
			inputShape,
			[]float64{1, 2, 3, 4, 5, 6},
		)

		Convey("It should transpose dimensions without staging through host dispatch", func() {
			output, err := tensorBackend.TransposeTensor(input, outputShape, 0, 1)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(output.Shape().Dims(), ShouldResemble, []int{3, 2})
			So(values, ShouldResemble, []float64{1, 4, 2, 5, 3, 6})
		})
	})
}

func TestTensorBackend_ViewAsHeadsTensor(test *testing.T) {
	Convey("Given a resident CUDA view_as_heads operation", test, func() {
		tensorBackend := newCUDATensorBackendForTest(test)
		inputShape := mustCUDAShape(test, []int{1, 2, 4})
		outputShape := mustCUDAShape(test, []int{1, 2, 2, 2})
		input := uploadCUDATensorForTest(
			test,
			tensorBackend,
			inputShape,
			[]float64{1, 2, 3, 4, 5, 6, 7, 8},
		)

		Convey("It should move head storage on the device", func() {
			output, err := tensorBackend.ViewAsHeadsTensor(input, outputShape, 1, 2, 2, 2)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2, 5, 6, 3, 4, 7, 8})
		})
	})
}

func TestTensorBackend_LastTokenTensor(test *testing.T) {
	Convey("Given a resident CUDA last_token operation", test, func() {
		tensorBackend := newCUDATensorBackendForTest(test)
		inputShape := mustCUDAShape(test, []int{2, 3, 2})
		outputShape := mustCUDAShape(test, []int{2, 2})
		input := uploadCUDATensorForTest(
			test,
			tensorBackend,
			inputShape,
			[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		)

		Convey("It should select the final sequence token on the device", func() {
			output, err := tensorBackend.LastTokenTensor(input, outputShape, 2, 3, 2)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{5, 6, 11, 12})
		})
	})
}

func BenchmarkTensorBackend_CopyTensor(benchmark *testing.B) {
	tensorBackend := newCUDATensorBackendForBenchmark(benchmark)
	inputShape := mustCUDAShape(benchmark, []int{1, 8192})
	outputShape := mustCUDAShape(benchmark, []int{8192, 1})
	input := uploadCUDATensorForTest(
		benchmark,
		tensorBackend,
		inputShape,
		make([]float64, inputShape.Len()),
	)

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := tensorBackend.CopyTensor(input, outputShape)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func newCUDATensorBackendForTest(test testing.TB) *TensorBackend {
	test.Helper()

	if err := Available(); err != nil {
		test.Skipf("CUDA tensor backend unavailable: %v", err)
	}

	tensorBackend := NewTensorBackend()
	test.Cleanup(func() {
		_ = tensorBackend.Close()
	})

	return tensorBackend
}

func newCUDATensorBackendForBenchmark(benchmark *testing.B) *TensorBackend {
	benchmark.Helper()

	if err := Available(); err != nil {
		benchmark.Skip(err)
	}

	tensorBackend := NewTensorBackend()
	benchmark.Cleanup(func() {
		_ = tensorBackend.Close()
	})

	return tensorBackend
}

func uploadCUDATensorForTest(
	test testing.TB,
	tensorBackend *TensorBackend,
	shape computetensor.Shape,
	values []float64,
) computetensor.Float64Tensor {
	test.Helper()

	tensorValue, err := tensorBackend.UploadFloat64(shape, values)

	if err != nil {
		test.Fatalf("UploadFloat64 failed: %v", err)
	}

	test.Cleanup(func() {
		_ = tensorValue.Close()
	})

	return tensorValue
}

func mustCUDAShape(test testing.TB, dims []int) computetensor.Shape {
	test.Helper()

	shape, err := computetensor.NewShape(dims)

	if err != nil {
		test.Fatalf("NewShape(%v): %v", dims, err)
	}

	return shape
}
