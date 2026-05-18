//go:build darwin && cgo

package metal

import (
	"testing"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func BenchmarkConvolutionOps_Conv1dTensor(benchmark *testing.B) {
	input, weight, bias, outputShape, convolutionOps, closeBackend := metalConvolutionBenchmarkTensors(
		benchmark,
		[]int{1, 1, 8192},
		[]int{1, 1, 1},
		[]int{1, 1, 8192},
	)
	defer closeBackend()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := convolutionOps.Conv1dTensor(
			input,
			weight,
			bias,
			outputShape,
			1,
			1,
			8192,
			1,
			1,
			1,
			0,
			1,
			1,
		)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkConvolutionOps_Conv2dTensor(benchmark *testing.B) {
	input, weight, bias, outputShape, convolutionOps, closeBackend := metalConvolutionBenchmarkTensors(
		benchmark,
		[]int{1, 1, 1, 8192},
		[]int{1, 1, 1, 1},
		[]int{1, 1, 1, 8192},
	)
	defer closeBackend()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := convolutionOps.Conv2dTensor(
			input,
			weight,
			bias,
			outputShape,
			1,
			1,
			1,
			8192,
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

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkConvolutionOps_Conv3dTensor(benchmark *testing.B) {
	input, weight, bias, outputShape, convolutionOps, closeBackend := metalConvolutionBenchmarkTensors(
		benchmark,
		[]int{1, 1, 1, 1, 8192},
		[]int{1, 1, 1, 1, 1},
		[]int{1, 1, 1, 1, 8192},
	)
	defer closeBackend()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := convolutionOps.Conv3dTensor(
			input,
			weight,
			bias,
			outputShape,
			1,
			1,
			1,
			1,
			8192,
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

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func BenchmarkConvolutionOps_ConvTranspose2dTensor(benchmark *testing.B) {
	input, weight, bias, outputShape, convolutionOps, closeBackend := metalConvolutionBenchmarkTensors(
		benchmark,
		[]int{1, 1, 1, 8192},
		[]int{1, 1, 1, 1},
		[]int{1, 1, 1, 8192},
	)
	defer closeBackend()

	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := convolutionOps.ConvTranspose2dTensor(
			input,
			weight,
			bias,
			outputShape,
			1,
			1,
			1,
			8192,
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
			0,
			0,
		)

		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func metalConvolutionBenchmarkTensors(
	benchmark *testing.B,
	inputDims []int,
	weightDims []int,
	outputDims []int,
) (
	inputTensor computetensor.Tensor,
	weightTensor computetensor.Tensor,
	biasTensor computetensor.Tensor,
	outputShape computetensor.Shape,
	convolutionOps *ConvolutionOps,
	closeBackend func(),
) {
	benchmark.Helper()

	lib := metallibPathOrSkip(benchmark, "convolution.metallib")
	tensorBackend, err := NewTensorBackend()

	if err != nil {
		benchmark.Skip(err)
	}

	convolutionOps, err = NewConvolutionOps(lib)
	if err != nil {
		benchmark.Fatal(err)
	}

	inputShape := mustMetalConvolutionShape(benchmark, inputDims)
	weightShape := mustMetalConvolutionShape(benchmark, weightDims)
	biasShape := mustMetalConvolutionShape(benchmark, []int{1})
	outputShape = mustMetalConvolutionShape(benchmark, outputDims)
	input := uploadMetalTensorForTest(
		benchmark,
		tensorBackend,
		inputShape,
		metalConvolutionSequence(8192, 0.017, -0.11),
	)
	weight := uploadMetalTensorForTest(benchmark, tensorBackend, weightShape, []float64{0.375})
	bias := uploadMetalTensorForTest(benchmark, tensorBackend, biasShape, []float64{-0.25})

	return input, weight, bias, outputShape, convolutionOps, func() {
		_ = tensorBackend.Close()
	}
}
