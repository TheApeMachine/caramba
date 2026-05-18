//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

var metalMathContractSizes = []int{1, 7, 64, 1024, 8192}

func TestMathOps_ExpTensor(test *testing.T) {
	Convey("Given resident Metal exp inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar reference at contract sizes", func() {
			for _, elementCount := range metalMathContractSizes {
				input := metalMathInput(elementCount, -2, 2)
				expected := mapFloat32Reference(input, math.Exp)

				output := runUnaryMathTensorForTest(test, tensorBackend, mathOps.ExpTensor, input)
				assertMetalMaxDiff(output, expected, 2e-5)
			}
		})
	})
}

func TestMathOps_LogTensor(test *testing.T) {
	Convey("Given resident Metal log inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should match the float32 scalar reference at contract sizes", func() {
			for _, elementCount := range metalMathContractSizes {
				input := positiveMetalMathInput(elementCount)
				expected := mapFloat32Reference(input, math.Log)

				output := runUnaryMathTensorForTest(test, tensorBackend, mathOps.LogTensor, input)
				assertMetalMaxDiff(output, expected, 2e-5)
			}
		})
	})
}

func TestMathOps_SignTensor(test *testing.T) {
	Convey("Given resident Metal sign inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should match the scalar sign contract at contract sizes", func() {
			for _, elementCount := range metalMathContractSizes {
				input := signedMetalMathInput(elementCount)
				expected := make([]float64, len(input))

				for index, value := range input {
					if value > 0 {
						expected[index] = 1
					}
					if value < 0 {
						expected[index] = -1
					}
				}

				output := runUnaryMathTensorForTest(test, tensorBackend, mathOps.SignTensor, input)
				So(output, ShouldResemble, expected)
			}
		})
	})
}

func TestMathOps_InvSqrtDimScaleTensor(test *testing.T) {
	Convey("Given resident Metal scale inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should scale by the inverse square root of the last dimension", func() {
			shape, err := computetensor.NewShape([]int{2, 4})
			So(err, ShouldBeNil)

			input := []float64{1, 2, 3, 4, -1, -2, -3, -4}
			expected := make([]float64, len(input))
			scale := 1 / math.Sqrt(4)

			for index, value := range input {
				expected[index] = float64(float32(float32(value) * float32(scale)))
			}

			tensor := uploadMetalTensorForTest(test, tensorBackend, shape, input)
			output, err := mathOps.InvSqrtDimScaleTensor(tensor)
			So(err, ShouldBeNil)
			defer func() {
				So(output.Close(), ShouldBeNil)
			}()

			values, err := tensorFloat64Values(output)
			So(err, ShouldBeNil)
			assertMetalMaxDiff(values, expected, 1e-6)
		})
	})
}

func TestMathOps_SoftmaxTensor(test *testing.T) {
	Convey("Given resident Metal softmax inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should match the scalar last-dimension reference", func() {
			for _, elementCount := range metalMathContractSizes {
				shape, input := rowMathInput(elementCount)
				expected := referenceSoftmaxRows(input, shape.Dims()[1])

				tensor := uploadMetalTensorForTest(test, tensorBackend, shape, input)
				output, err := mathOps.SoftmaxTensor(tensor)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				assertMetalMaxDiff(values, expected, 2e-5)
			}
		})
	})
}

func TestMathOps_LogSumExpTensor(test *testing.T) {
	Convey("Given resident Metal logsumexp inputs", test, func() {
		tensorBackend := newMetalTensorBackendForTest(test)
		mathOps := metalMathOpsForTest(test, tensorBackend)

		Convey("It should reduce over the last dimension", func() {
			for _, elementCount := range metalMathContractSizes {
				shape, input := rowMathInput(elementCount)
				expected := referenceLogSumExpRows(input, shape.Dims()[1])

				tensor := uploadMetalTensorForTest(test, tensorBackend, shape, input)
				output, err := mathOps.LogSumExpTensor(tensor)
				So(err, ShouldBeNil)
				defer func() {
					So(output.Close(), ShouldBeNil)
				}()

				values, err := tensorFloat64Values(output)
				So(err, ShouldBeNil)
				So(output.Location(), ShouldEqual, computetensor.Metal)
				So(output.Shape().Dims(), ShouldResemble, []int{len(expected)})
				assertMetalMaxDiff(values, expected, 2e-5)
			}
		})
	})
}

func BenchmarkMathOps_ExpTensor(benchmark *testing.B) {
	benchmarkUnaryMathTensor(benchmark, func(mathOps *MathOps) mathTensorUnary {
		return mathOps.ExpTensor
	})
}

func BenchmarkMathOps_LogTensor(benchmark *testing.B) {
	benchmarkUnaryMathTensor(benchmark, func(mathOps *MathOps) mathTensorUnary {
		return mathOps.LogTensor
	})
}

func BenchmarkMathOps_SignTensor(benchmark *testing.B) {
	benchmarkUnaryMathTensor(benchmark, func(mathOps *MathOps) mathTensorUnary {
		return mathOps.SignTensor
	})
}

func BenchmarkMathOps_InvSqrtDimScaleTensor(benchmark *testing.B) {
	benchmarkUnaryMathTensor(benchmark, func(mathOps *MathOps) mathTensorUnary {
		return mathOps.InvSqrtDimScaleTensor
	})
}

func BenchmarkMathOps_SoftmaxTensor(benchmark *testing.B) {
	benchmarkReductionMathTensor(benchmark, func(mathOps *MathOps) mathTensorUnary {
		return mathOps.SoftmaxTensor
	})
}

func BenchmarkMathOps_LogSumExpTensor(benchmark *testing.B) {
	benchmarkReductionMathTensor(benchmark, func(mathOps *MathOps) mathTensorUnary {
		return mathOps.LogSumExpTensor
	})
}

type mathTensorUnary func(computetensor.Tensor) (computetensor.Tensor, error)

func metalMathOpsForTest(test testing.TB, tensorBackend *TensorBackend) *MathOps {
	test.Helper()

	mathOps, err := tensorBackend.math()
	So(err, ShouldBeNil)

	return mathOps
}

func runUnaryMathTensorForTest(
	test testing.TB,
	tensorBackend *TensorBackend,
	operation mathTensorUnary,
	input []float64,
) []float64 {
	test.Helper()

	shape, err := computetensor.NewShape([]int{len(input)})
	So(err, ShouldBeNil)

	tensor := uploadMetalTensorForTest(test, tensorBackend, shape, input)
	output, err := operation(tensor)
	So(err, ShouldBeNil)
	defer func() {
		So(output.Close(), ShouldBeNil)
	}()

	values, err := tensorFloat64Values(output)
	So(err, ShouldBeNil)
	So(output.Location(), ShouldEqual, computetensor.Metal)

	return values
}

func benchmarkUnaryMathTensor(
	benchmark *testing.B,
	selectOperation func(*MathOps) mathTensorUnary,
) {
	benchmarkMathTensor(benchmark, []int{8192}, metalMathInput(8192, 0.25, 1.25), selectOperation)
}

func benchmarkReductionMathTensor(
	benchmark *testing.B,
	selectOperation func(*MathOps) mathTensorUnary,
) {
	benchmarkMathTensor(benchmark, []int{128, 64}, metalMathInput(8192, -1, 1), selectOperation)
}

func benchmarkMathTensor(
	benchmark *testing.B,
	dimensions []int,
	input []float64,
	selectOperation func(*MathOps) mathTensorUnary,
) {
	benchmark.ReportAllocs()

	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	mathOps, err := tensorBackend.math()
	if err != nil {
		benchmark.Fatal(err)
	}

	shape, err := computetensor.NewShape(dimensions)
	if err != nil {
		benchmark.Fatal(err)
	}

	tensor := uploadMetalTensor(tensorBackend, shape, input)
	defer func() {
		_ = tensor.Close()
	}()

	operation := selectOperation(mathOps)
	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := operation(tensor)
		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func metalMathInput(elementCount int, low, high float64) []float64 {
	values := make([]float64, elementCount)
	span := high - low

	for index := range values {
		values[index] = low + span*float64(index%257)/256
	}

	return values
}

func positiveMetalMathInput(elementCount int) []float64 {
	values := make([]float64, elementCount)

	for index := range values {
		values[index] = 0.125 + float64(index%257)/32
	}

	return values
}

func signedMetalMathInput(elementCount int) []float64 {
	values := make([]float64, elementCount)

	for index := range values {
		values[index] = float64(index%7 - 3)
	}

	return values
}

func mapFloat32Reference(input []float64, operation func(float64) float64) []float64 {
	output := make([]float64, len(input))

	for index, value := range input {
		output[index] = float64(float32(operation(float64(float32(value)))))
	}

	return output
}

func rowMathInput(elementCount int) (computetensor.Shape, []float64) {
	shape, err := computetensor.NewShape([]int{1, elementCount})
	if err != nil {
		panic(err)
	}

	return shape, metalMathInput(elementCount, -2, 2)
}

func referenceSoftmaxRows(input []float64, dimSize int) []float64 {
	output := make([]float64, len(input))

	for offset := 0; offset < len(input); offset += dimSize {
		writeSoftmaxRow(output[offset:offset+dimSize], input[offset:offset+dimSize])
	}

	return output
}

func writeSoftmaxRow(output []float64, input []float64) {
	maxValue := input[0]

	for _, value := range input[1:] {
		maxValue = math.Max(maxValue, value)
	}

	sum := 0.0
	for index, value := range input {
		output[index] = math.Exp(float64(float32(value - maxValue)))
		sum += output[index]
	}

	for index := range output {
		output[index] = float64(float32(output[index] / sum))
	}
}

func referenceLogSumExpRows(input []float64, dimSize int) []float64 {
	output := make([]float64, len(input)/dimSize)

	for row := range output {
		output[row] = referenceLogSumExp(input[row*dimSize : (row+1)*dimSize])
	}

	return output
}

func referenceLogSumExp(input []float64) float64 {
	maxValue := input[0]

	for _, value := range input[1:] {
		maxValue = math.Max(maxValue, value)
	}

	sum := 0.0
	for _, value := range input {
		sum += math.Exp(float64(float32(value - maxValue)))
	}

	return float64(float32(math.Log(sum) + maxValue))
}
