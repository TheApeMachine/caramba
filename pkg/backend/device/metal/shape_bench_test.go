package metal

import (
	"context"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func BenchmarkKernel_RunShapeDTypes(benchmark *testing.B) {
	backend := newBackendForBenchmark(benchmark)
	defer func() {
		_ = backend.Close()
	}()

	for _, storageDType := range metalShapeDTypes {
		storageDType := storageDType

		benchmark.Run(storageDType.Name(), func(benchmark *testing.B) {
			for _, testCase := range metalShapeCases() {
				testCase := testCase

				benchmark.Run(testCase.name, func(benchmark *testing.B) {
					run, outputs, byteCount, closeAll := shapeBenchmarkSetup(
						benchmark,
						backend,
						storageDType,
						testCase.name,
					)
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

func shapeBenchmarkSetup(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	name string,
) (func() error, []tensor.Tensor, int64, func()) {
	switch name {
	case "reshape":
		return unaryShapeBenchmarkSetup(testingObject, backend, storageDType, name, []int{8192}, []int{1, 8192})
	case "merge_heads":
		return unaryShapeBenchmarkSetup(testingObject, backend, storageDType, name, []int{1, 8192, 2, 3}, []int{1, 8192, 6})
	case "split_heads":
		return unaryShapeBenchmarkSetup(testingObject, backend, storageDType, name, []int{1, 8192, 6}, []int{1, 8192, 2, 3})
	case "view_as_heads":
		return viewAsHeadsBenchmarkSetup(testingObject, backend, storageDType)
	case "concat":
		return concatShapeBenchmarkSetup(testingObject, backend, storageDType)
	case "split2":
		return split2ShapeBenchmarkSetup(testingObject, backend, storageDType)
	case "last_token":
		return unaryShapeBenchmarkSetup(testingObject, backend, storageDType, name, []int{2, 3, 8192}, []int{2, 8192})
	case "transpose2d":
		return unaryShapeBenchmarkSetup(testingObject, backend, storageDType, name, []int{8192, 2}, []int{2, 8192})
	case "upsample_nearest2d":
		return unaryShapeBenchmarkSetup(testingObject, backend, storageDType, name, []int{1, 1, 8192, 2}, []int{1, 1, 16384, 4})
	}

	testingObject.Fatalf("unknown shape benchmark: %s", name)
	return nil, nil, 0, nil
}

func unaryShapeBenchmarkSetup(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	name string,
	inputDims []int,
	outDims []int,
) (func() error, []tensor.Tensor, int64, func()) {
	inputShape := mustShapeForTest(testingObject, inputDims)
	outShape := mustShapeForTest(testingObject, outDims)
	inputBytes := rawShapeBytesForTest(testingObject, inputShape, storageDType)
	kernel := lookupUnaryShapeKernel(testingObject, name, storageDType)
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	byteCount := int64(len(inputBytes) + out.Bytes())

	return func() error { return kernel.Run(input, out) },
		[]tensor.Tensor{out},
		byteCount,
		func() { closeBenchmarkTensors(input, out) }
}

func viewAsHeadsBenchmarkSetup(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) (func() error, []tensor.Tensor, int64, func()) {
	inputShape := mustShapeForTest(testingObject, []int{1, 8192, 6})
	outShape := mustShapeForTest(testingObject, []int{1, 8192, 2, 3})
	inputBytes := rawShapeBytesForTest(testingObject, inputShape, storageDType)
	kernel := lookupViewAsHeadsKernel(testingObject, storageDType)
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	heads := uploadInt32ScalarForTest(testingObject, backend, 2)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	byteCount := int64(len(inputBytes) + out.Bytes())

	return func() error { return kernel.Run(input, heads, out) },
		[]tensor.Tensor{out},
		byteCount,
		func() { closeBenchmarkTensors(input, heads, out) }
}

func concatShapeBenchmarkSetup(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) (func() error, []tensor.Tensor, int64, func()) {
	leftShape := mustShapeForTest(testingObject, []int{8192})
	rightShape := mustShapeForTest(testingObject, []int{8192})
	outShape := mustShapeForTest(testingObject, []int{16384})
	leftBytes := rawShapeBytesForTest(testingObject, leftShape, storageDType)
	rightBytes := shiftedRawBytesForTest(testingObject, rightShape, storageDType, 19)
	kernel := lookupBinaryShapeKernel(testingObject, "concat", storageDType)
	left := uploadDTypeTensorForTest(testingObject, backend, leftShape, storageDType, leftBytes)
	right := uploadDTypeTensorForTest(testingObject, backend, rightShape, storageDType, rightBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	byteCount := int64(len(leftBytes) + len(rightBytes) + out.Bytes())

	return func() error { return kernel.Run(left, right, out) },
		[]tensor.Tensor{out},
		byteCount,
		func() { closeBenchmarkTensors(left, right, out) }
}

func split2ShapeBenchmarkSetup(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
) (func() error, []tensor.Tensor, int64, func()) {
	inputShape := mustShapeForTest(testingObject, []int{16384})
	outShape := mustShapeForTest(testingObject, []int{8192})
	inputBytes := rawShapeBytesForTest(testingObject, inputShape, storageDType)
	kernel := lookupSplit2Kernel(testingObject, storageDType)
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	left := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	right := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	byteCount := int64(len(inputBytes) + left.Bytes() + right.Bytes())

	return func() error { return kernel.Run(input, left, right) },
		[]tensor.Tensor{left, right},
		byteCount,
		func() { closeBenchmarkTensors(input, left, right) }
}
