package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalMathDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalMathDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalMathDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalMathDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" tensors for math utilities", testingObject, func() {
				runInvSqrtDimScaleParityCase(testingObject, backend, storageDType, elementCount)
				runLogSumExpParityCase(testingObject, backend, storageDType, elementCount)
				runOuterParityCase(testingObject, backend, storageDType, elementCount)
			})
		})
	}
}

func runInvSqrtDimScaleParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	scaleDim := int32(17)
	fixture := invSqrtDimScaleFixtureForTest(elementCount, storageDType, scaleDim)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.inputBytes)
	dim := uploadInt32ScalarForTest(testingObject, backend, scaleDim)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(input, dim, out)

	err := lookupInvSqrtDimScaleKernel(testingObject, storageDType).Run(input, dim, out)
	convey.So(err, convey.ShouldBeNil)
	assertMathUnaryBytesForTest(
		testingObject, backend, out, storageDType, fixture, mathMaxULP("inv_sqrt_dim_scale", storageDType),
	)
}

func runLogSumExpParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	cols int,
) {
	rows := 7
	fixture := logSumExpFixtureForTest(rows, cols, storageDType)
	inputShape := mustShapeForTest(testingObject, []int{rows, cols})
	outShape := mustShapeForTest(testingObject, []int{rows})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, fixture.inputBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(input, out)

	err := lookupLogSumExpKernel(testingObject, storageDType).Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertMathUnaryBytesForTest(
		testingObject, backend, out, storageDType, fixture, mathMaxULP("logsumexp", storageDType),
	)
}

func runOuterParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	cols int,
) {
	rows := 7
	fixture := outerFixtureForTest(rows, cols, storageDType)
	leftShape := mustShapeForTest(testingObject, []int{rows})
	rightShape := mustShapeForTest(testingObject, []int{cols})
	outShape := mustShapeForTest(testingObject, []int{rows, cols})
	left := uploadDTypeTensorForTest(testingObject, backend, leftShape, storageDType, fixture.leftBytes)
	right := uploadDTypeTensorForTest(testingObject, backend, rightShape, storageDType, fixture.rightBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(left, right, out)

	err := lookupOuterKernel(testingObject, storageDType).Run(left, right, out)
	convey.So(err, convey.ShouldBeNil)
	assertMathOuterBytesForTest(
		testingObject, backend, out, storageDType, fixture, mathMaxULP("outer", storageDType),
	)
}

func lookupInvSqrtDimScaleKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("inv_sqrt_dim_scale", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, dtype.Int32},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s inv_sqrt_dim_scale kernel", storageDType.Name())
	}

	return kernel
}

func lookupLogSumExpKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("logsumexp", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s logsumexp kernel", storageDType.Name())
	}

	return kernel
}

func lookupOuterKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("outer", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s outer kernel", storageDType.Name())
	}

	return kernel
}

func assertMathUnaryBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	fixture mathUnaryFixture,
	maxULP uint32,
) {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, input, storageDType, fixture.expectedBytes, maxULP)
		return
	}

	assertFloat32TensorForTest(testingObject, backend, input, fixture.expectedFloat32, maxULP)
}

func assertMathOuterBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	fixture mathOuterFixture,
	maxULP uint32,
) {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, input, storageDType, fixture.expectedBytes, maxULP)
		return
	}

	assertFloat32TensorForTest(testingObject, backend, input, fixture.expectedFloat32, maxULP)
}

func mathMaxULP(name string, storageDType dtype.DType) uint32 {
	if storageDType != dtype.Float32 {
		return 2
	}

	if name == "logsumexp" {
		return 64
	}

	return 4
}
