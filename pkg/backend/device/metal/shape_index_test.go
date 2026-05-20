package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalShapeIndexDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalShapeDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalShapeIndexDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalShapeIndexDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" shape-index tensors", testingObject, func() {
				runGatherParityCase(testingObject, backend, storageDType, elementCount)
				runScatterParityCase(testingObject, backend, storageDType, elementCount)
				runWhereParityCase(testingObject, backend, storageDType, elementCount)
				runMaskedFillParityCase(testingObject, backend, storageDType, elementCount)
				runTransposeParityCase(testingObject, backend, storageDType, elementCount)
			})
		})
	}
}

func runGatherParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	inner := 3
	fixture := gatherFixtureForTest(elementCount, inner, storageDType)
	sourceShape := mustShapeForTest(testingObject, []int{elementCount + 5, inner})
	indexShape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{elementCount, inner})
	source := uploadDTypeTensorForTest(testingObject, backend, sourceShape, storageDType, fixture.firstBytes)
	indices := uploadDTypeTensorForTest(testingObject, backend, indexShape, dtype.Int32, fixture.secondBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(source, indices, out)

	err := lookupGatherKernel(testingObject, storageDType).Run(source, indices, out)
	convey.So(err, convey.ShouldBeNil)
	assertShapeIndexOutput(testingObject, backend, out, storageDType, fixture)
}

func runScatterParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	inner := 3
	fixture := scatterFixtureForTest(elementCount, inner, storageDType)
	targetShape := mustShapeForTest(testingObject, []int{elementCount + 5, inner})
	indexShape := mustShapeForTest(testingObject, []int{elementCount})
	updateShape := mustShapeForTest(testingObject, []int{elementCount, inner})
	target := uploadDTypeTensorForTest(testingObject, backend, targetShape, storageDType, fixture.firstBytes)
	indices := uploadDTypeTensorForTest(testingObject, backend, indexShape, dtype.Int32, fixture.secondBytes)
	updates := uploadDTypeTensorForTest(testingObject, backend, updateShape, storageDType, fixture.thirdBytes)
	out := emptyTensorForTest(testingObject, backend, targetShape, storageDType)
	defer closeBenchmarkTensors(target, indices, updates, out)

	err := lookupScatterKernel(testingObject, storageDType).Run(target, indices, updates, out)
	convey.So(err, convey.ShouldBeNil)
	assertShapeIndexOutput(testingObject, backend, out, storageDType, fixture)
}

func runWhereParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := whereFixtureForTest(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	mask := uploadDTypeTensorForTest(testingObject, backend, shape, dtype.Bool, fixture.firstBytes)
	positive := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.secondBytes)
	negative := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.thirdBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(mask, positive, negative, out)

	err := lookupWhereKernel(testingObject, storageDType).Run(mask, positive, negative, out)
	convey.So(err, convey.ShouldBeNil)
	assertShapeIndexOutput(testingObject, backend, out, storageDType, fixture)
}

func runMaskedFillParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := maskedFillFixtureForTest(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	scalarShape := mustShapeForTest(testingObject, []int{1})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.firstBytes)
	mask := uploadDTypeTensorForTest(testingObject, backend, shape, dtype.Bool, fixture.secondBytes)
	scalar := uploadDTypeTensorForTest(testingObject, backend, scalarShape, storageDType, fixture.thirdBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(input, mask, scalar, out)

	err := lookupMaskedFillKernel(testingObject, storageDType).Run(input, mask, scalar, out)
	convey.So(err, convey.ShouldBeNil)
	assertShapeIndexOutput(testingObject, backend, out, storageDType, fixture)
}

func runTransposeParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := transposeFixtureForTest(elementCount, storageDType)
	inputShape := mustShapeForTest(testingObject, transposeInputDims(elementCount))
	permShape := mustShapeForTest(testingObject, []int{len(fixture.permutation)})
	outShape := mustShapeForTest(testingObject, transposeOutputDims(elementCount))
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, fixture.firstBytes)
	permutation := uploadDTypeTensorForTest(testingObject, backend, permShape, dtype.Int32, fixture.secondBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(input, permutation, out)

	err := lookupTransposeKernel(testingObject, storageDType).Run(input, permutation, out)
	convey.So(err, convey.ShouldBeNil)
	assertShapeIndexOutput(testingObject, backend, out, storageDType, fixture)
}

func lookupGatherKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	return lookupShapeIndexKernel(testingObject, "gather", []dtype.DType{storageDType, dtype.Int32}, storageDType)
}

func lookupScatterKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	return lookupShapeIndexKernel(
		testingObject, "scatter", []dtype.DType{storageDType, dtype.Int32, storageDType}, storageDType,
	)
}

func lookupWhereKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	return lookupShapeIndexKernel(
		testingObject, "where", []dtype.DType{dtype.Bool, storageDType, storageDType}, storageDType,
	)
}

func lookupMaskedFillKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	return lookupShapeIndexKernel(
		testingObject, "masked_fill", []dtype.DType{storageDType, dtype.Bool, storageDType}, storageDType,
	)
}

func lookupTransposeKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	return lookupShapeIndexKernel(testingObject, "transpose", []dtype.DType{storageDType, dtype.Int32}, storageDType)
}

func lookupShapeIndexKernel(
	testingObject testing.TB,
	name string,
	inputs []dtype.DType,
	output dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  inputs,
		Outputs: []dtype.DType{output},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", output.Name(), name)
	}

	return kernel
}

func assertShapeIndexOutput(
	testingObject testing.TB,
	backend *Backend,
	out tensor.Tensor,
	storageDType dtype.DType,
	fixture shapeIndexFixture,
) {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, out, storageDType, fixture.expectedBytes, 0)
		return
	}

	assertFloat32TensorForTest(testingObject, backend, out, fixture.expectedValues, 0)
}
