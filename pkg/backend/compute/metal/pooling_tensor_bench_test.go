//go:build darwin && cgo

package metal

import (
	"testing"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func BenchmarkPoolingOps_MaxPool2dTensor(benchmark *testing.B) {
	benchmarkPool2dTensor(benchmark, func(
		poolingOps *PoolingOps,
		outputShape computetensor.Shape,
	) poolTensorOperation {
		return func(input computetensor.Tensor) (computetensor.Tensor, error) {
			return poolingOps.MaxPool2dTensor(input, outputShape, maxPoolTestParams())
		}
	})
}

func BenchmarkPoolingOps_AvgPool2dTensor(benchmark *testing.B) {
	benchmarkPool2dTensor(benchmark, func(
		poolingOps *PoolingOps,
		outputShape computetensor.Shape,
	) poolTensorOperation {
		return func(input computetensor.Tensor) (computetensor.Tensor, error) {
			return poolingOps.AvgPool2dTensor(input, outputShape, avgPoolTestParams())
		}
	})
}

func BenchmarkPoolingOps_AdaptiveAvgPool2dTensor(benchmark *testing.B) {
	benchmarkAdaptivePool2dTensor(benchmark, func(
		poolingOps *PoolingOps,
		outputShape computetensor.Shape,
	) poolTensorOperation {
		return func(input computetensor.Tensor) (computetensor.Tensor, error) {
			return poolingOps.AdaptiveAvgPool2dTensor(input, outputShape)
		}
	})
}

func BenchmarkPoolingOps_AdaptiveMaxPool2dTensor(benchmark *testing.B) {
	benchmarkAdaptivePool2dTensor(benchmark, func(
		poolingOps *PoolingOps,
		outputShape computetensor.Shape,
	) poolTensorOperation {
		return func(input computetensor.Tensor) (computetensor.Tensor, error) {
			return poolingOps.AdaptiveMaxPool2dTensor(input, outputShape)
		}
	})
}

func benchmarkPool2dTensor(
	benchmark *testing.B,
	operation poolBenchmarkOperation,
) {
	inputShape, err := computetensor.NewShape([]int{4, 16, 32, 32})
	if err != nil {
		benchmark.Fatal(err)
	}

	outputShape, err := computetensor.NewShape([]int{4, 16, 32, 32})
	if err != nil {
		benchmark.Fatal(err)
	}

	benchmarkPoolingTensor(benchmark, inputShape, outputShape, operation)
}

func benchmarkAdaptivePool2dTensor(
	benchmark *testing.B,
	operation poolBenchmarkOperation,
) {
	inputShape, err := computetensor.NewShape([]int{4, 16, 32, 32})
	if err != nil {
		benchmark.Fatal(err)
	}

	outputShape, err := computetensor.NewShape([]int{4, 16, 8, 8})
	if err != nil {
		benchmark.Fatal(err)
	}

	benchmarkPoolingTensor(benchmark, inputShape, outputShape, operation)
}

type poolBenchmarkOperation func(*PoolingOps, computetensor.Shape) poolTensorOperation

func benchmarkPoolingTensor(
	benchmark *testing.B,
	inputShape computetensor.Shape,
	outputShape computetensor.Shape,
	operation poolBenchmarkOperation,
) {
	benchmark.ReportAllocs()

	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}
	defer func() {
		_ = tensorBackend.Close()
	}()

	poolingOps, err := tensorBackend.pooling()
	if err != nil {
		benchmark.Fatal(err)
	}

	input := uploadMetalTensor(tensorBackend, inputShape, poolingSequence(inputShape.Len()))
	defer func() {
		_ = input.Close()
	}()

	apply := operation(poolingOps, outputShape)
	benchmark.ResetTimer()

	for benchmark.Loop() {
		output, err := apply(input)
		if err != nil {
			benchmark.Fatal(err)
		}

		if err := output.Close(); err != nil {
			benchmark.Fatal(err)
		}
	}
}
