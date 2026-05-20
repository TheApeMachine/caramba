package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunLossDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalLossDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			benchmarkLossDType(benchmark, backend, storageDType)
		})
	}
}

func benchmarkLossDType(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, name := range metalPairLossNames {
		name := name

		benchmark.Run(name, func(benchmark *testing.B) {
			benchmarkPairLoss(benchmark, backend, name, storageDType)
		})
	}

	benchmark.Run("cross_entropy", func(benchmark *testing.B) {
		benchmarkCrossEntropyLoss(benchmark, backend, storageDType)
	})
}

func benchmarkPairLoss(
	benchmark *testing.B,
	backend *Backend,
	name string,
	storageDType dtype.DType,
) {
	elementCount := 8192
	predictions, targets, out := pairLossBenchmarkTensors(
		benchmark, backend, name, storageDType, elementCount,
	)
	defer closeBenchmarkTensors(predictions, targets, out)

	benchmark.SetBytes(pairLossBenchmarkBytes(elementCount, storageDType))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := lookupPairLossKernel(benchmark, name, storageDType).Run(predictions, targets, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func benchmarkCrossEntropyLoss(
	benchmark *testing.B,
	backend *Backend,
	storageDType dtype.DType,
) {
	batch, classes := 64, 1024
	logits, targets, out := crossEntropyLossBenchmarkTensors(
		benchmark, backend, storageDType, batch, classes,
	)
	defer closeBenchmarkTensors(logits, targets, out)

	benchmark.SetBytes(crossEntropyLossBenchmarkBytes(batch, classes, storageDType))
	benchmark.ResetTimer()

	for benchmark.Loop() {
		if err := lookupCrossEntropyLossKernel(benchmark, storageDType).Run(logits, targets, out); err != nil {
			benchmark.Fatal(err)
		}

		if err := out.Sync(context.Background()); err != nil {
			benchmark.Fatal(err)
		}
	}
}

func pairLossBenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
	elementCount int,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	fixture := lossPairFixtureForTest(name, elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{1})
	predictions := uploadDTypeTensorForTest(
		testingObject, backend, shape, storageDType, fixture.predictionBytes,
	)
	targets := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.targetBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return predictions, targets, out
}

func crossEntropyLossBenchmarkTensors(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	batch int,
	classes int,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	logits := lossCrossEntropyLogits(batch, classes)
	targets := lossCrossEntropyTargets(batch, classes)
	logitShape := mustShapeForTest(testingObject, []int{batch, classes})
	targetShape := mustShapeForTest(testingObject, []int{batch})
	outShape := mustShapeForTest(testingObject, []int{1})
	logitBytes := encodeLossValuesAsDType(logits, storageDType)
	logitTensor := uploadDTypeTensorForTest(testingObject, backend, logitShape, storageDType, logitBytes)
	targetTensor := uploadDTypeTensorForTest(
		testingObject, backend, targetShape, dtype.Int32, int32ValuesToBytes(targets),
	)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)

	return logitTensor, targetTensor, out
}

func pairLossBenchmarkBytes(elementCount int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	return int64(elementCount)*elementBytes*2 + elementBytes
}

func crossEntropyLossBenchmarkBytes(batch int, classes int, storageDType dtype.DType) int64 {
	elementBytes := int64(dtypeBytesForBenchmark(storageDType))
	logitBytes := int64(batch*classes) * elementBytes
	targetBytes := int64(batch * 4)

	return logitBytes + targetBytes + elementBytes
}
