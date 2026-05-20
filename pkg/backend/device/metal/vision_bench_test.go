package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunVisionDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalVisionDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkConv1DDType(benchmark, backend, storageDType)
			benchmarkConv2DDType(benchmark, backend, storageDType)
			benchmarkConv3DDType(benchmark, backend, storageDType)
			benchmarkConvTranspose2DDType(benchmark, backend, storageDType)
			benchmarkPool2DDType(benchmark, backend, storageDType)
			benchmarkAdaptivePool2DDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkConv1DDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("conv1d", func(benchmark *testing.B) {
		width := 1024
		input, weight, bias, out := benchmarkConv1DTensors(benchmark, backend, width, storageDType)
		defer closeBenchmarkTensors(input, weight, bias, out)

		benchmark.SetBytes(conv1DBenchmarkBytes(width, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalConv1D(input, weight, bias, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkConv2DDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("conv2d", func(benchmark *testing.B) {
		outputWidth := 512
		input, weight, bias, out := benchmarkConv2DTensors(
			benchmark, backend, outputWidth, storageDType,
		)
		defer closeBenchmarkTensors(input, weight, bias, out)

		benchmark.SetBytes(conv2DBenchmarkBytes(outputWidth, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalConv2D(input, weight, bias, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkConv3DDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("conv3d", func(benchmark *testing.B) {
		width := 512
		input, weight, bias, out := benchmarkConv3DTensors(benchmark, backend, width, storageDType)
		defer closeBenchmarkTensors(input, weight, bias, out)

		benchmark.SetBytes(conv3DBenchmarkBytes(width, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalConv3D(input, weight, bias, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkConvTranspose2DDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("conv_transpose2d", func(benchmark *testing.B) {
		width := 512
		input, weight, bias, out := benchmarkConvTranspose2DTensors(
			benchmark, backend, width, storageDType,
		)
		defer closeBenchmarkTensors(input, weight, bias, out)

		benchmark.SetBytes(convTranspose2DBenchmarkBytes(width, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalConvTranspose2D(input, weight, bias, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkPool2DDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("max_pool2d", func(benchmark *testing.B) {
		input, out := benchmarkPool2DTensors(benchmark, backend, 1024, storageDType)
		defer closeBenchmarkTensors(input, out)
		benchmark.SetBytes(pool2DBenchmarkBytes(1024, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalMaxPool2D(input, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})

	benchmark.Run("avg_pool2d", func(benchmark *testing.B) {
		input, out := benchmarkPool2DTensors(benchmark, backend, 1024, storageDType)
		defer closeBenchmarkTensors(input, out)
		benchmark.SetBytes(pool2DBenchmarkBytes(1024, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalAvgPool2D(input, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkAdaptivePool2DDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	benchmark.Run("adaptive_avg_pool2d", func(benchmark *testing.B) {
		input, out := benchmarkAdaptivePool2DTensors(benchmark, backend, 1024, storageDType)
		defer closeBenchmarkTensors(input, out)
		benchmark.SetBytes(adaptivePool2DBenchmarkBytes(1024, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalAdaptiveAvgPool2D(input, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})

	benchmark.Run("adaptive_max_pool2d", func(benchmark *testing.B) {
		input, out := benchmarkAdaptivePool2DTensors(benchmark, backend, 1024, storageDType)
		defer closeBenchmarkTensors(input, out)
		benchmark.SetBytes(adaptivePool2DBenchmarkBytes(1024, storageDType))
		benchmark.ResetTimer()

		for benchmark.Loop() {
			if err := runMetalAdaptiveMaxPool2D(input, out); err != nil {
				benchmark.Fatal(err)
			}

			if err := out.Sync(context.Background()); err != nil {
				benchmark.Fatal(err)
			}
		}
	})
}

func benchmarkConv2DTensors(
	testingObject testing.TB,
	backend *Backend,
	outputWidth int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputBytes, weightBytes, biasBytes, _ := conv2DDTypeBytes(outputWidth, storageDType)
	return conv2DTensorsForTest(
		testingObject, backend, outputWidth, storageDType, inputBytes, weightBytes, biasBytes,
	)
}

func benchmarkConv1DTensors(
	testingObject testing.TB,
	backend *Backend,
	width int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputBytes, weightBytes, biasBytes, _ := conv1DDTypeBytes(width, storageDType)
	return conv1DTensorsForTest(
		testingObject, backend, width, storageDType, inputBytes, weightBytes, biasBytes,
	)
}

func benchmarkConv3DTensors(
	testingObject testing.TB,
	backend *Backend,
	width int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputBytes, weightBytes, biasBytes, _ := conv3DDTypeBytes(width, storageDType)
	return conv3DTensorsForTest(
		testingObject, backend, width, storageDType, inputBytes, weightBytes, biasBytes,
	)
}

func benchmarkConvTranspose2DTensors(
	testingObject testing.TB,
	backend *Backend,
	width int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputBytes, weightBytes, biasBytes, _ := convTranspose2DDTypeBytes(width, storageDType)
	return convTranspose2DTensorsForTest(
		testingObject, backend, width, storageDType, inputBytes, weightBytes, biasBytes,
	)
}

func benchmarkPool2DTensors(
	testingObject testing.TB,
	backend *Backend,
	outputWidth int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputBytes, _, _ := pool2DDTypeBytes(outputWidth, storageDType)
	input, maxOut, _ := pool2DTensorsForTest(testingObject, backend, outputWidth, storageDType, inputBytes)
	return input, maxOut
}

func benchmarkAdaptivePool2DTensors(
	testingObject testing.TB,
	backend *Backend,
	outputWidth int,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	inputBytes, _, _ := adaptivePool2DDTypeBytes(outputWidth, storageDType)
	input, avgOut, _ := adaptivePool2DTensorsForTest(
		testingObject, backend, outputWidth, storageDType, inputBytes,
	)
	return input, avgOut
}

func conv2DBenchmarkBytes(outputWidth int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	batch, inChannels, outChannels := 2, 2, 3
	inputHeight, kernelHeight, kernelWidth := 4, 2, 3
	inputWidth := outputWidth + kernelWidth - 1
	outputHeight := inputHeight - kernelHeight + 1
	elements := batch*inChannels*inputHeight*inputWidth +
		outChannels*inChannels*kernelHeight*kernelWidth +
		outChannels +
		batch*outChannels*outputHeight*outputWidth
	return int64(elements) * elementBytes
}

func conv1DBenchmarkBytes(width int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	batch, inChannels, outChannels, kernelLength := 2, 2, 3, 3
	inputLength := width + kernelLength - 1
	elements := batch*inChannels*inputLength +
		outChannels*inChannels*kernelLength +
		outChannels +
		batch*outChannels*width
	return int64(elements) * elementBytes
}

func conv3DBenchmarkBytes(width int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	batch, inChannels, outChannels := 1, 2, 2
	inputDepth, inputHeight, kernelDepth, kernelHeight, kernelWidth := 3, 3, 2, 2, 3
	inputWidth := width + kernelWidth - 1
	elements := batch*inChannels*inputDepth*inputHeight*inputWidth +
		outChannels*inChannels*kernelDepth*kernelHeight*kernelWidth +
		outChannels +
		batch*outChannels*2*2*width
	return int64(elements) * elementBytes
}

func convTranspose2DBenchmarkBytes(width int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	batch, inChannels, outChannels := 2, 2, 3
	inputHeight, kernelHeight, kernelWidth := 3, 2, 3
	elements := batch*inChannels*inputHeight*width +
		inChannels*outChannels*kernelHeight*kernelWidth +
		outChannels +
		batch*outChannels*(inputHeight+kernelHeight-1)*(width+kernelWidth-1)
	return int64(elements) * elementBytes
}

func pool2DBenchmarkBytes(outputWidth int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	batch, channels, inputHeight := 2, 3, 4
	inputWidth := outputWidth * 2
	outputHeight := inputHeight / 2
	elements := batch*channels*inputHeight*inputWidth +
		batch*channels*outputHeight*outputWidth
	return int64(elements) * elementBytes
}

func adaptivePool2DBenchmarkBytes(outputWidth int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	batch, channels, inputHeight, outputHeight := 2, 2, 5, 3
	inputWidth := outputWidth + 3
	elements := batch*channels*inputHeight*inputWidth +
		batch*channels*outputHeight*outputWidth
	return int64(elements) * elementBytes
}
