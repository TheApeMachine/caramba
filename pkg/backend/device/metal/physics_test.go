package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestKernelRegistry_MetalPhysicsDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalPhysicsDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalPhysicsDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalPhysicsDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" physics tensors", testingObject, func() {
				runPhysicsUnaryParityCases(testingObject, backend, storageDType, elementCount)
				runMadelungParityCase(testingObject, backend, storageDType, elementCount)
				runFFTParityCase(testingObject, backend, storageDType, elementCount, false)
				runFFTParityCase(testingObject, backend, storageDType, elementCount, true)
			})
		})
	}
}

func runPhysicsUnaryParityCases(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	for _, name := range physicsUnaryNames() {
		runPhysicsUnaryParityCase(testingObject, backend, storageDType, name, elementCount)
	}
}

func runPhysicsUnaryParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	name string,
	elementCount int,
) {
	dims := physicsDimsForTest(name, elementCount)
	fixture := physicsUnaryFixtureForTest(name, dims, storageDType)
	shape := mustShapeForTest(testingObject, dims)
	input := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.inputBytes)
	spacing := uploadDTypeTensorForTest(
		testingObject, backend, scalarShapeForTest(testingObject), storageDType, fixture.spacingBytes,
	)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(input, spacing, out)

	err := lookupPhysicsBinaryKernel(testingObject, name, storageDType).Run(input, spacing, out)
	convey.So(err, convey.ShouldBeNil)
	assertPhysicsOutput(testingObject, backend, out, storageDType, fixture.expectedBytes, fixture.expectedFloat32, name)
}

func runMadelungParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := madelungFixtureForTest(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	density := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.firstBytes)
	velocity := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.secondBytes)
	spacing := uploadDTypeTensorForTest(
		testingObject, backend, scalarShapeForTest(testingObject), storageDType, fixture.spacingBytes,
	)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(density, velocity, spacing, out)

	err := lookupMadelungKernel(testingObject, storageDType).Run(density, velocity, spacing, out)
	convey.So(err, convey.ShouldBeNil)
	assertPhysicsOutput(
		testingObject, backend, out, storageDType, fixture.expectedBytes,
		fixture.expectedFloat32, "madelung_continuity",
	)
}

func runFFTParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
	inverse bool,
) {
	fixture := physicsFFTFixtureForTest(elementCount, storageDType, inverse)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	realIn := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.realInBytes)
	imagIn := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.imagInBytes)
	realOut := emptyTensorForTest(testingObject, backend, shape, storageDType)
	imagOut := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(realIn, imagIn, realOut, imagOut)

	name := "fft1d"
	if inverse {
		name = "ifft1d"
	}

	err := lookupPhysicsFFTKernel(testingObject, name, storageDType).Run(realIn, imagIn, realOut, imagOut)
	convey.So(err, convey.ShouldBeNil)
	assertPhysicsOutput(testingObject, backend, realOut, storageDType, fixture.expectedRealBytes, fixture.expectedReal, name)
	assertPhysicsOutput(testingObject, backend, imagOut, storageDType, fixture.expectedImagBytes, fixture.expectedImag, name)
}

func physicsUnaryNames() []string {
	return []string{
		"laplacian",
		"laplacian4",
		"grad1d",
		"divergence1d",
		"quantum_potential",
		"bohmian_velocity",
	}
}

func physicsDimsForTest(name string, elementCount int) []int {
	if name != "laplacian" {
		return []int{elementCount}
	}

	switch elementCount {
	case 64:
		return []int{8, 8}
	case 1024:
		return []int{32, 32}
	case 8192:
		return []int{8, 32, 32}
	default:
		return []int{elementCount}
	}
}

func lookupPhysicsBinaryKernel(testingObject testing.TB, name string, storageDType dtype.DType) kernels.Kernel {
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

func lookupMadelungKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("madelung_continuity", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType,
			storageDType,
			storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s madelung_continuity kernel", storageDType.Name())
	}

	return kernel
}

func lookupPhysicsFFTKernel(testingObject testing.TB, name string, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, storageDType},
		Outputs: []dtype.DType{storageDType, storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", storageDType.Name(), name)
	}

	return kernel
}

func assertPhysicsOutput(
	testingObject testing.TB,
	backend *Backend,
	out tensor.Tensor,
	storageDType dtype.DType,
	expectedBytes []byte,
	expectedFloat32 []float32,
	name string,
) {
	testingObject.Helper()
	maxULP := physicsMaxULP(name, storageDType)

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, out, storageDType, expectedBytes, maxULP)
		return
	}

	assertFloat32TensorForTest(testingObject, backend, out, expectedFloat32, maxULP)
}

func physicsMaxULP(name string, storageDType dtype.DType) uint32 {
	if storageDType != dtype.Float32 {
		return 2
	}

	switch name {
	case "quantum_potential":
		return 96
	case "fft1d", "ifft1d":
		return 4096
	default:
		return 8
	}
}
