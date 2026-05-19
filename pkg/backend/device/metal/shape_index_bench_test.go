package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func BenchmarkKernel_RunShapeIndexDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalShapeDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			for _, name := range []string{"gather", "scatter", "where", "masked_fill", "transpose"} {
				name := name

				benchmark.Run(name, func(benchmark *testing.B) {
					run, outputs, byteCount, closeAll := shapeIndexBenchmarkSetup(benchmark, backend, storageDType, name)
					defer closeAll()

					benchmark.SetBytes(byteCount)
					benchmark.ResetTimer()

					for benchmark.Loop() {
						if err := run(); err != nil {
							benchmark.Fatal(err)
						}

						for _, output := range outputs {
							if err := output.Sync(context.Background()); err != nil {
								benchmark.Fatal(err)
							}
						}
					}
				})
			}
		})
	}
}

func shapeIndexBenchmarkSetup(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	name string,
) (func() error, []tensor.Tensor, int64, func()) {
	switch name {
	case "gather":
		return gatherShapeIndexBenchmarkSetup(testingObject, backend, storageDType)
	case "scatter":
		return scatterShapeIndexBenchmarkSetup(testingObject, backend, storageDType)
	case "where":
		return whereShapeIndexBenchmarkSetup(testingObject, backend, storageDType)
	case "masked_fill":
		return maskedFillShapeIndexBenchmarkSetup(testingObject, backend, storageDType)
	case "transpose":
		return transposeShapeIndexBenchmarkSetup(testingObject, backend, storageDType)
	}

	testingObject.Fatalf("unknown shape-index benchmark: %s", name)
	return nil, nil, 0, nil
}

func gatherShapeIndexBenchmarkSetup(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) (func() error, []tensor.Tensor, int64, func()) {
	elementCount := 8192
	inner := 4
	fixture := gatherFixtureForTest(elementCount, inner, storageDType)
	sourceShape := mustShapeForTest(testingObject, []int{elementCount + 5, inner})
	indexShape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{elementCount, inner})
	source := uploadDTypeTensorForTest(testingObject, backend, sourceShape, storageDType, fixture.firstBytes)
	indices := uploadDTypeTensorForTest(testingObject, backend, indexShape, dtype.Int32, fixture.secondBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	byteCount := int64(len(fixture.firstBytes) + len(fixture.secondBytes) + out.Bytes())

	return func() error { return lookupGatherKernel(testingObject, storageDType).Run(source, indices, out) },
		[]tensor.Tensor{out},
		byteCount,
		func() { closeBenchmarkTensors(source, indices, out) }
}

func scatterShapeIndexBenchmarkSetup(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) (func() error, []tensor.Tensor, int64, func()) {
	elementCount := 8192
	inner := 4
	fixture := scatterFixtureForTest(elementCount, inner, storageDType)
	targetShape := mustShapeForTest(testingObject, []int{elementCount + 5, inner})
	indexShape := mustShapeForTest(testingObject, []int{elementCount})
	updateShape := mustShapeForTest(testingObject, []int{elementCount, inner})
	target := uploadDTypeTensorForTest(testingObject, backend, targetShape, storageDType, fixture.firstBytes)
	indices := uploadDTypeTensorForTest(testingObject, backend, indexShape, dtype.Int32, fixture.secondBytes)
	updates := uploadDTypeTensorForTest(testingObject, backend, updateShape, storageDType, fixture.thirdBytes)
	out := emptyTensorForTest(testingObject, backend, targetShape, storageDType)
	byteCount := int64(len(fixture.firstBytes) + len(fixture.secondBytes) + len(fixture.thirdBytes) + out.Bytes())

	return func() error {
			return lookupScatterKernel(testingObject, storageDType).Run(target, indices, updates, out)
		},
		[]tensor.Tensor{out},
		byteCount,
		func() { closeBenchmarkTensors(target, indices, updates, out) }
}

func whereShapeIndexBenchmarkSetup(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) (func() error, []tensor.Tensor, int64, func()) {
	elementCount := 8192
	fixture := whereFixtureForTest(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	mask := uploadDTypeTensorForTest(testingObject, backend, shape, dtype.Bool, fixture.firstBytes)
	positive := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.secondBytes)
	negative := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.thirdBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	byteCount := int64(len(fixture.firstBytes) + len(fixture.secondBytes) + len(fixture.thirdBytes) + out.Bytes())

	return func() error { return lookupWhereKernel(testingObject, storageDType).Run(mask, positive, negative, out) },
		[]tensor.Tensor{out},
		byteCount,
		func() { closeBenchmarkTensors(mask, positive, negative, out) }
}

func maskedFillShapeIndexBenchmarkSetup(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) (func() error, []tensor.Tensor, int64, func()) {
	elementCount := 8192
	fixture := maskedFillFixtureForTest(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	scalarShape := mustShapeForTest(testingObject, []int{1})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.firstBytes)
	mask := uploadDTypeTensorForTest(testingObject, backend, shape, dtype.Bool, fixture.secondBytes)
	scalar := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.thirdBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	byteCount := int64(len(fixture.firstBytes) + len(fixture.secondBytes) + len(fixture.thirdBytes) + out.Bytes())

	return func() error { return lookupMaskedFillKernel(testingObject, storageDType).Run(input, mask, scalar, out) },
		[]tensor.Tensor{out},
		byteCount,
		func() { closeBenchmarkTensors(input, mask, scalar, out) }
}

func transposeShapeIndexBenchmarkSetup(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) (func() error, []tensor.Tensor, int64, func()) {
	elementCount := 8192
	fixture := transposeFixtureForTest(elementCount, storageDType)
	inputShape := mustShapeForTest(testingObject, transposeInputDims(elementCount))
	permShape := mustShapeForTest(testingObject, []int{len(fixture.permutation)})
	outShape := mustShapeForTest(testingObject, transposeOutputDims(elementCount))
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, fixture.firstBytes)
	permutation := uploadDTypeTensorForTest(testingObject, backend, permShape, dtype.Int32, fixture.secondBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	byteCount := int64(len(fixture.firstBytes) + len(fixture.secondBytes) + out.Bytes())

	return func() error { return lookupTransposeKernel(testingObject, storageDType).Run(input, permutation, out) },
		[]tensor.Tensor{out},
		byteCount,
		func() { closeBenchmarkTensors(input, permutation, out) }
}
