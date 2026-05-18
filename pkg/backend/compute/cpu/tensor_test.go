package cpu

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
	dtypeconvert "github.com/theapemachine/caramba/pkg/dtype/convert"
)

func TestNewTensorBackend(t *testing.T) {
	Convey("Given a CPU tensor backend", t, func() {
		tensorBackend := NewTensorBackend()

		Convey("It should report host residency", func() {
			So(tensorBackend.Location(), ShouldEqual, computetensor.Host)
		})
	})
}

func TestTensorBackend_Upload(t *testing.T) {
	Convey("Given a CPU tensor backend", t, func() {
		tensorBackend := NewTensorBackend()

		Convey("It should upload values into host residency", func() {
			tensorValue := uploadTestTensor(tensorBackend, []int{3}, []float64{1, 2, 3})
			defer func() { So(tensorValue.Close(), ShouldBeNil) }()

			So(tensorValue.Location(), ShouldEqual, computetensor.Host)
			So(tensorValue.Shape().Dims(), ShouldResemble, []int{3})
			So(tensorValue.Len(), ShouldEqual, 3)
		})
	})
}

func TestTensorBackend_Download(t *testing.T) {
	Convey("Given a resident CPU tensor", t, func() {
		tensorBackend := NewTensorBackend()
		tensorValue := uploadTestTensor(tensorBackend, []int{3}, []float64{1, 2, 3})
		defer func() { So(tensorValue.Close(), ShouldBeNil) }()

		Convey("It should download the stored values", func() {
			values, err := tensorFloat64Values(tensorValue)

			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{1, 2, 3})
		})
	})
}

func TestTensorBackend_Close(t *testing.T) {
	Convey("Given a closed CPU tensor backend", t, func() {
		tensorBackend := NewTensorBackend()
		err := tensorBackend.Close()
		So(err, ShouldBeNil)

		Convey("It should reject uploads", func() {
			shape, err := computetensor.NewShape([]int{1})
			So(err, ShouldBeNil)

			output, err := tensorBackend.Upload(
				shape,
				dtype.Float64,
				dtypeconvert.Float64ToBytes([]float64{1}),
			)

			So(err, ShouldNotBeNil)
			So(output, ShouldBeNil)
		})
	})
}

func BenchmarkTensorBackend_Upload(benchmark *testing.B) {
	tensorBackend := NewTensorBackend()
	values := make([]float64, 64*64)

	for index := range values {
		values[index] = float64(index)
	}

	shape, err := computetensor.NewShape([]int{64, 64})

	if err != nil {
		benchmark.Fatal(err)
	}

	bytes := dtypeconvert.Float64ToBytes(values)

	for benchmark.Loop() {
		output, err := tensorBackend.Upload(shape, dtype.Float64, bytes)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func uploadTestTensor(
	tensorBackend *TensorBackend, dims []int, values []float64,
) computetensor.Tensor {
	shape, err := computetensor.NewShape(dims)
	So(err, ShouldBeNil)

	input, err := tensorBackend.Upload(
		shape,
		dtype.Float64,
		dtypeconvert.Float64ToBytes(values),
	)
	So(err, ShouldBeNil)

	return input
}

func uploadBenchmarkTensor(
	benchmark *testing.B,
	tensorBackend *TensorBackend,
	dims []int,
	values []float64,
) computetensor.Tensor {
	shape, err := computetensor.NewShape(dims)

	if err != nil {
		benchmark.Fatal(err)
	}

	input, err := tensorBackend.Upload(
		shape,
		dtype.Float64,
		dtypeconvert.Float64ToBytes(values),
	)

	if err != nil {
		benchmark.Fatal(err)
	}

	return input
}

func tensorFloat64Values(value computetensor.Tensor) ([]float64, error) {
	sourceDType, bytes, err := value.RawBytes()

	if err != nil {
		return nil, err
	}

	return dtypeconvert.BytesToFloat64(sourceDType, bytes)
}
