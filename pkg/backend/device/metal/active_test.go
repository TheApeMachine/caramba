package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestKernelRegistry_MetalActiveInferenceDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalActiveDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalActiveInferenceDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalActiveInferenceDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" tensors for active inference", testingObject, func() {
				runFreeEnergyParityCase(testingObject, backend, storageDType, elementCount)
				runExpectedFreeEnergyParityCase(testingObject, backend, storageDType, elementCount)
				runActiveBinaryParityCase(testingObject, backend, "belief_update", storageDType, elementCount)
				runActiveBinaryParityCase(testingObject, backend, "precision_weight", storageDType, elementCount)
			})
		})
	}
}

func runFreeEnergyParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := activeFreeEnergyFixtureForTest(storageDType, elementCount)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{1})
	likelihood := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.firstBytes)
	posterior := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.secondBytes)
	prior := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.thirdBytes)
	auxiliary := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.auxiliaryBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(likelihood, posterior, prior, auxiliary, out)

	err := lookupActiveFreeEnergyKernel(testingObject, storageDType).Run(
		likelihood, posterior, prior, auxiliary, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertActiveBytesForTest(testingObject, backend, out, storageDType, fixture, activeScalarMaxULP(storageDType))
}

func runExpectedFreeEnergyParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := activeExpectedFreeEnergyFixtureForTest(storageDType, elementCount, elementCount)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{1})
	predictedObs := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.firstBytes)
	preferredObs := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.secondBytes)
	predictedState := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.thirdBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(predictedObs, preferredObs, predictedState, out)

	err := lookupActiveExpectedFreeEnergyKernel(testingObject, storageDType).Run(
		predictedObs, preferredObs, predictedState, out,
	)
	convey.So(err, convey.ShouldBeNil)
	assertActiveBytesForTest(testingObject, backend, out, storageDType, fixture, activeScalarMaxULP(storageDType))
}

func runActiveBinaryParityCase(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := activeBinaryFixtureForTest(name, storageDType, elementCount)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	left := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.firstBytes)
	right := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.secondBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(left, right, out)

	err := lookupActiveBinaryKernel(testingObject, name, storageDType).Run(left, right, out)
	convey.So(err, convey.ShouldBeNil)
	assertActiveBytesForTest(testingObject, backend, out, storageDType, fixture, activeBinaryMaxULP(name, storageDType))
}

func lookupActiveFreeEnergyKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("free_energy", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType,
			storageDType,
			storageDType,
			storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s free_energy kernel", storageDType.Name())
	}

	return kernel
}

func lookupActiveExpectedFreeEnergyKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("expected_free_energy", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType,
			storageDType,
			storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s expected_free_energy kernel", storageDType.Name())
	}

	return kernel
}

func lookupActiveBinaryKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", storageDType.Name(), name)
	}

	return kernel
}

func assertActiveBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	fixture activeFixture,
	maxULP uint32,
) {
	testingObject.Helper()

	if storageDType == dtype.Float32 {
		assertFloat32TensorForTest(testingObject, backend, input, fixture.expectedFloat32, maxULP)
		return
	}

	assertDTypeBytesForTest(testingObject, backend, input, storageDType, fixture.expectedBytes, maxULP)
}

func activeScalarMaxULP(storageDType dtype.DType) uint32 {
	if storageDType == dtype.Float32 {
		return 256
	}

	return 2
}

func activeBinaryMaxULP(name string, storageDType dtype.DType) uint32 {
	if storageDType != dtype.Float32 {
		return 2
	}

	if name == "belief_update" {
		return 16
	}

	return 2
}
