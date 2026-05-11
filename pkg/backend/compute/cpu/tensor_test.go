package cpu

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestNewTensorBackend(t *testing.T) {
	Convey("Given a CPU tensor backend", t, func() {
		tensorBackend := NewTensorBackend()

		Convey("It should report host residency", func() {
			So(tensorBackend.Location(), ShouldEqual, computetensor.Host)
		})
	})
}

func TestTensorBackend_ReLU(t *testing.T) {
	Convey("Given a resident CPU tensor", t, func() {
		tensorBackend := NewTensorBackend()
		input := uploadTestTensor(tensorBackend, []int{4}, []float64{-2, -0, 3, 4})

		Convey("It should execute ReLU without changing residency", func() {
			output, err := tensorBackend.ReLU(input)
			So(err, ShouldBeNil)
			So(output.Location(), ShouldEqual, computetensor.Host)

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{0, 0, 3, 4})
		})
	})
}

func TestTensorBackend_LeakyReLU(t *testing.T) {
	Convey("Given a resident CPU tensor", t, func() {
		tensorBackend := NewTensorBackend()
		input := uploadTestTensor(tensorBackend, []int{3}, []float64{-2, 0, 4})

		Convey("It should execute parameterized ReLU", func() {
			output, err := tensorBackend.LeakyReLU(input, 0.25)
			So(err, ShouldBeNil)

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{-0.5, 0, 4})
		})
	})
}

func TestTensorBackend_Sigmoid(t *testing.T) {
	Convey("Given a resident CPU tensor", t, func() {
		tensorBackend := NewTensorBackend()
		input := uploadTestTensor(tensorBackend, []int{3}, []float64{-1, 0, 1})

		Convey("It should execute sigmoid", func() {
			output, err := tensorBackend.Sigmoid(input)
			So(err, ShouldBeNil)

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values[0], ShouldBeLessThan, values[1])
			So(values[1], ShouldAlmostEqual, 0.5, 1e-9)
			So(values[2], ShouldBeGreaterThan, values[1])
		})
	})
}

func TestTensorBackend_SwiGLU(t *testing.T) {
	Convey("Given a resident CPU tensor with doubled final dimension", t, func() {
		tensorBackend := NewTensorBackend()
		input := uploadTestTensor(tensorBackend, []int{1, 4}, []float64{0, 1, 2, 4})

		Convey("It should halve the final dimension and stay resident", func() {
			output, err := tensorBackend.SwiGLU(input)
			So(err, ShouldBeNil)
			So(output.Shape().Dims(), ShouldResemble, []int{1, 2})
			So(output.Location(), ShouldEqual, computetensor.Host)

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values[0], ShouldAlmostEqual, 1.0, 1e-9)
			So(values[1], ShouldBeGreaterThan, 2.0)
		})
	})
}

func TestTensorBackend_Add(t *testing.T) {
	Convey("Given two resident CPU tensors", t, func() {
		tensorBackend := NewTensorBackend()
		left := uploadTestTensor(tensorBackend, []int{3}, []float64{1, 2, 3})
		right := uploadTestTensor(tensorBackend, []int{3}, []float64{4, 5, 6})

		Convey("It should add without downloading inputs", func() {
			output, err := tensorBackend.Add(left, right)
			So(err, ShouldBeNil)

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{5, 7, 9})
		})
	})
}

func TestTensorBackend_Mul(t *testing.T) {
	Convey("Given two resident CPU tensors", t, func() {
		tensorBackend := NewTensorBackend()
		left := uploadTestTensor(tensorBackend, []int{3}, []float64{1, 2, 3})
		right := uploadTestTensor(tensorBackend, []int{3}, []float64{4, 5, 6})

		Convey("It should multiply without downloading inputs", func() {
			output, err := tensorBackend.Mul(left, right)
			So(err, ShouldBeNil)

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{4, 10, 18})
		})
	})
}

func TestTensorBackend_Matmul(t *testing.T) {
	Convey("Given resident CPU matrices", t, func() {
		tensorBackend := NewTensorBackend()
		left := uploadTestTensor(tensorBackend, []int{2, 3}, []float64{
			1, 2, 3,
			4, 5, 6,
		})
		right := uploadTestTensor(tensorBackend, []int{3, 2}, []float64{
			7, 8,
			9, 10,
			11, 12,
		})

		Convey("It should multiply using the CPU matmul kernel", func() {
			output, err := tensorBackend.Matmul(left, right)
			So(err, ShouldBeNil)
			So(output.Shape().Dims(), ShouldResemble, []int{2, 2})

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{58, 64, 139, 154})
		})
	})
}

func TestTensorBackend_MatmulAdd(t *testing.T) {
	Convey("Given resident CPU matrices and bias", t, func() {
		tensorBackend := NewTensorBackend()
		left := uploadTestTensor(tensorBackend, []int{2, 3}, []float64{
			1, 2, 3,
			4, 5, 6,
		})
		right := uploadTestTensor(tensorBackend, []int{3, 2}, []float64{
			7, 8,
			9, 10,
			11, 12,
		})
		bias := uploadTestTensor(tensorBackend, []int{2}, []float64{1, -1})

		Convey("It should fuse matmul and broadcast bias", func() {
			output, err := tensorBackend.MatmulAdd(left, right, bias)
			So(err, ShouldBeNil)
			So(output.Shape().Dims(), ShouldResemble, []int{2, 2})

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values, ShouldResemble, []float64{59, 63, 140, 153})
		})
	})
}

func TestTensorBackend_MatmulAddGELU(t *testing.T) {
	Convey("Given resident CPU matrices and bias", t, func() {
		tensorBackend := NewTensorBackend()
		left := uploadTestTensor(tensorBackend, []int{1, 2}, []float64{1, -1})
		right := uploadTestTensor(tensorBackend, []int{2, 2}, []float64{
			1, 2,
			3, 4,
		})
		bias := uploadTestTensor(tensorBackend, []int{2}, []float64{0, 1})

		Convey("It should fuse matmul, bias, and GELU", func() {
			output, err := tensorBackend.MatmulAddGELU(left, right, bias)
			So(err, ShouldBeNil)

			values, err := output.CloneFloat64()
			So(err, ShouldBeNil)
			So(values[0], ShouldAlmostEqual, -0.0454023059, 1e-9)
			So(values[1], ShouldAlmostEqual, -0.1588080094, 1e-9)
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

			output, err := tensorBackend.UploadFloat64(shape, []float64{1})

			So(err, ShouldNotBeNil)
			So(output, ShouldBeNil)
		})
	})
}

func BenchmarkTensorBackend_Matmul(benchmark *testing.B) {
	tensorBackend := NewTensorBackend()
	leftValues := make([]float64, 64*64)
	rightValues := make([]float64, 64*64)

	left := uploadBenchmarkTensor(benchmark, tensorBackend, []int{64, 64}, leftValues)
	right := uploadBenchmarkTensor(benchmark, tensorBackend, []int{64, 64}, rightValues)

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		output, err := tensorBackend.Matmul(left, right)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkTensorBackend_MatmulAddGELU(benchmark *testing.B) {
	tensorBackend := NewTensorBackend()
	leftValues := make([]float64, 64*64)
	rightValues := make([]float64, 64*64)
	biasValues := make([]float64, 64)

	left := uploadBenchmarkTensor(benchmark, tensorBackend, []int{64, 64}, leftValues)
	right := uploadBenchmarkTensor(benchmark, tensorBackend, []int{64, 64}, rightValues)
	bias := uploadBenchmarkTensor(benchmark, tensorBackend, []int{64}, biasValues)

	benchmark.ResetTimer()

	for iteration := 0; iteration < benchmark.N; iteration++ {
		output, err := tensorBackend.MatmulAddGELU(left, right, bias)

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
) computetensor.Float64Tensor {
	shape, err := computetensor.NewShape(dims)
	So(err, ShouldBeNil)

	input, err := tensorBackend.UploadFloat64(shape, values)
	So(err, ShouldBeNil)

	return input
}

func uploadBenchmarkTensor(
	benchmark *testing.B,
	tensorBackend *TensorBackend,
	dims []int,
	values []float64,
) computetensor.Float64Tensor {
	shape, err := computetensor.NewShape(dims)

	if err != nil {
		benchmark.Fatal(err)
	}

	input, err := tensorBackend.UploadFloat64(shape, values)

	if err != nil {
		benchmark.Fatal(err)
	}

	return input
}
