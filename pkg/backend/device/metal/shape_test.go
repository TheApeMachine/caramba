package metal

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestKernelRegistry_MetalShapeDTypes(t *testing.T) {
	backend := newBackendForDeviceTest(t)
	defer func() {
		if err := backend.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalShapeDTypes {
		storageDType := storageDType

		t.Run(storageDType.Name(), func(t *testing.T) {
			testMetalShapeDType(t, backend, storageDType)
		})
	}
}

func testMetalShapeDType(
	t *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, testCase := range metalShapeCases() {
		testCase := testCase

		t.Run(testCase.name, func(t *testing.T) {
			for _, elementCount := range parityElementCounts {
				elementCount := elementCount

				t.Run(fmt.Sprintf("N=%d", elementCount), func(t *testing.T) {
					convey.Convey("Given Metal "+storageDType.Name()+" tensors for "+testCase.name, t, func() {
						testCase.run(t, backend, storageDType, elementCount)
					})
				})
			}
		})
	}
}

type metalShapeCase struct {
	name string
	run  func(testing.TB, *Backend, dtype.DType, int)
}

func metalShapeCases() []metalShapeCase {
	return []metalShapeCase{
		{name: "reshape", run: runMetalReshapeShapeCase},
		{name: "merge_heads", run: runMetalMergeHeadsShapeCase},
		{name: "split_heads", run: runMetalSplitHeadsShapeCase},
		{name: "view_as_heads", run: runMetalViewAsHeadsShapeCase},
		{name: "concat", run: runMetalConcatShapeCase},
		{name: "split2", run: runMetalSplit2ShapeCase},
		{name: "last_token", run: runMetalLastTokenShapeCase},
		{name: "transpose2d", run: runMetalTranspose2DShapeCase},
		{name: "upsample_nearest2d", run: runMetalUpsampleNearest2DShapeCase},
	}
}

func runMetalReshapeShapeCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	inputShape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{1, elementCount})
	inputBytes := rawShapeBytesForTest(testingObject, inputShape, storageDType)
	kernel := lookupUnaryShapeKernel(testingObject, "reshape", storageDType)
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	defer closeBenchmarkTensors(input)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(out)

	convey.So(kernel.Run(input, out), convey.ShouldBeNil)
	assertRawBytesForTest(testingObject, backend, out, storageDType, inputBytes)
}

func runMetalMergeHeadsShapeCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	inputShape := mustShapeForTest(testingObject, []int{1, elementCount, 2, 3})
	outShape := mustShapeForTest(testingObject, []int{1, elementCount, 6})
	runUnaryRawShapeCase(testingObject, backend, "merge_heads", storageDType, inputShape, outShape)
}

func runMetalSplitHeadsShapeCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	inputShape := mustShapeForTest(testingObject, []int{1, elementCount, 6})
	outShape := mustShapeForTest(testingObject, []int{1, elementCount, 2, 3})
	runUnaryRawShapeCase(testingObject, backend, "split_heads", storageDType, inputShape, outShape)
}

func runMetalViewAsHeadsShapeCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	inputShape := mustShapeForTest(testingObject, []int{1, elementCount, 6})
	outShape := mustShapeForTest(testingObject, []int{1, elementCount, 2, 3})
	inputBytes := rawShapeBytesForTest(testingObject, inputShape, storageDType)
	kernel := lookupViewAsHeadsKernel(testingObject, storageDType)
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	defer closeBenchmarkTensors(input)
	heads := uploadInt32ScalarForTest(testingObject, backend, 2)
	defer closeBenchmarkTensors(heads)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(out)

	convey.So(kernel.Run(input, heads, out), convey.ShouldBeNil)
	assertRawBytesForTest(testingObject, backend, out, storageDType, inputBytes)
}

func runMetalConcatShapeCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	leftShape := mustShapeForTest(testingObject, []int{elementCount})
	rightShape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{2 * elementCount})
	leftBytes := rawShapeBytesForTest(testingObject, leftShape, storageDType)
	rightBytes := shiftedRawBytesForTest(testingObject, rightShape, storageDType, 19)
	expectedBytes := append(append([]byte(nil), leftBytes...), rightBytes...)
	kernel := lookupBinaryShapeKernel(testingObject, "concat", storageDType)
	left := uploadDTypeTensorForTest(testingObject, backend, leftShape, storageDType, leftBytes)
	defer closeBenchmarkTensors(left)
	right := uploadDTypeTensorForTest(testingObject, backend, rightShape, storageDType, rightBytes)
	defer closeBenchmarkTensors(right)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(out)

	convey.So(kernel.Run(left, right, out), convey.ShouldBeNil)
	assertRawBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func runMetalSplit2ShapeCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	inputShape := mustShapeForTest(testingObject, []int{2 * elementCount})
	outShape := mustShapeForTest(testingObject, []int{elementCount})
	inputBytes := rawShapeBytesForTest(testingObject, inputShape, storageDType)
	leftBytes, rightBytes := splitRawBytesForTest(inputBytes)
	kernel := lookupSplit2Kernel(testingObject, storageDType)
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	defer closeBenchmarkTensors(input)
	left := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(left)
	right := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(right)

	convey.So(kernel.Run(input, left, right), convey.ShouldBeNil)
	assertRawBytesForTest(testingObject, backend, left, storageDType, leftBytes)
	assertRawBytesForTest(testingObject, backend, right, storageDType, rightBytes)
}

func runMetalLastTokenShapeCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	inputShape := mustShapeForTest(testingObject, []int{2, 3, elementCount})
	outShape := mustShapeForTest(testingObject, []int{2, elementCount})
	inputBytes := rawShapeBytesForTest(testingObject, inputShape, storageDType)
	expectedBytes := lastTokenRawBytesForTest(testingObject, inputBytes, storageDType, 2, 3, elementCount)
	kernel := lookupUnaryShapeKernel(testingObject, "last_token", storageDType)
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	defer closeBenchmarkTensors(input)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(out)

	convey.So(kernel.Run(input, out), convey.ShouldBeNil)
	assertRawBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func runMetalTranspose2DShapeCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	inputShape := mustShapeForTest(testingObject, []int{elementCount, 2})
	outShape := mustShapeForTest(testingObject, []int{2, elementCount})
	inputBytes := rawShapeBytesForTest(testingObject, inputShape, storageDType)
	expectedBytes := transpose2DRawBytesForTest(testingObject, inputBytes, storageDType, elementCount, 2)
	kernel := lookupUnaryShapeKernel(testingObject, "transpose2d", storageDType)
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	defer closeBenchmarkTensors(input)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(out)

	convey.So(kernel.Run(input, out), convey.ShouldBeNil)
	assertRawBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func runMetalUpsampleNearest2DShapeCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	inputShape := mustShapeForTest(testingObject, []int{1, 1, elementCount, 2})
	outShape := mustShapeForTest(testingObject, []int{1, 1, elementCount * 2, 4})
	inputBytes := rawShapeBytesForTest(testingObject, inputShape, storageDType)
	expectedBytes := upsampleNearest2DRawBytesForTest(
		testingObject,
		inputBytes,
		storageDType,
		[]int{1, 1, elementCount, 2},
		[]int{1, 1, elementCount * 2, 4},
	)
	kernel := lookupUnaryShapeKernel(testingObject, "upsample_nearest2d", storageDType)
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	defer closeBenchmarkTensors(input)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(out)

	convey.So(kernel.Run(input, out), convey.ShouldBeNil)
	assertRawBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}

func runUnaryRawShapeCase(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
	inputShape tensor.Shape,
	outShape tensor.Shape,
) {
	inputBytes := rawShapeBytesForTest(testingObject, inputShape, storageDType)
	kernel := lookupUnaryShapeKernel(testingObject, name, storageDType)
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, inputBytes)
	defer closeBenchmarkTensors(input)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(out)

	convey.So(kernel.Run(input, out), convey.ShouldBeNil)
	assertRawBytesForTest(testingObject, backend, out, storageDType, inputBytes)
}
