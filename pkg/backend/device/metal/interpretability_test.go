package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestKernelRegistry_MetalInterpretabilityKernels(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal tensors for interpretability kernels", testingObject, func() {
				runActivationSteerParityCase(testingObject, backend, elementCount)
			})
		})
	}
}

func runActivationSteerParityCase(
	testingObject testing.TB,
	backend *Backend,
	elementCount int,
) {
	fixture := activationSteerFixtureForTest(elementCount)
	vectorShape := mustShapeForTest(testingObject, []int{elementCount})
	coefficientShape := mustShapeForTest(testingObject, []int{1})
	base := uploadDTypeTensorForTest(testingObject, backend, vectorShape, dtype.Float32, fixture.baseBytes)
	direction := uploadDTypeTensorForTest(testingObject, backend, vectorShape, dtype.Float32, fixture.directionBytes)
	coefficient := uploadDTypeTensorForTest(
		testingObject, backend, coefficientShape, dtype.Float32, fixture.coefficientBytes,
	)
	out := emptyTensorForTest(testingObject, backend, vectorShape, dtype.Float32)
	defer closeBenchmarkTensors(base, direction, coefficient, out)

	err := lookupActivationSteerFloat32Kernel(testingObject).Run(base, direction, coefficient, out)
	convey.So(err, convey.ShouldBeNil)
	assertUtilityBytesForTest(testingObject, backend, out, dtype.Float32, fixture.expectedBytes)
}

func lookupActivationSteerFloat32Kernel(testingObject testing.TB) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("activation_steer_float32", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			dtype.Float32,
			dtype.Float32,
			dtype.Float32,
		},
		Outputs: []dtype.DType{dtype.Float32},
	}, tensor.Metal)

	if !ok {
		testingObject.Fatalf("activation_steer_float32 Metal kernel not registered")
	}

	return kernel
}
