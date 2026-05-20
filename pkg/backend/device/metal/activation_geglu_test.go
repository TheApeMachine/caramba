package metal

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

const geGLUMaxULP uint32 = 2

func TestKernelRegistry_MetalGeGLUDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalGeGLUDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			for _, elementCount := range parityElementCounts {
				elementCount := elementCount

				testingObject.Run(fmt.Sprintf("N=%d", elementCount), func(testingObject *testing.T) {
					convey.Convey(
						"Given Metal "+storageDType.Name()+" tensors for geglu",
						testingObject,
						func() {
							runGeGLUParityCase(testingObject, backend, storageDType, elementCount)
						},
					)
				})
			}
		})
	}
}

func runGeGLUParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := geGLUFixtureForTest(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	gate := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.gateBytes)
	up := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.upBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(gate, up, out)

	err := lookupGeGLUKernel(testingObject, storageDType).Run(gate, up, out)
	convey.So(err, convey.ShouldBeNil)

	if storageDType == dtype.Float32 {
		actualDType, actualBytes, downloadErr := backend.Download(out)
		convey.So(downloadErr, convey.ShouldBeNil)
		convey.So(actualDType, convey.ShouldEqual, dtype.Float32)
		assertFloat32WithinULP(
			testingObject,
			mustFloat32Bytes(actualBytes),
			mustFloat32Bytes(fixture.expectedBytes),
			geGLUMaxULP,
		)
		return
	}

	assertDTypeBytesForTest(testingObject, backend, out, storageDType, fixture.expectedBytes, geGLUMaxULP)
}

func lookupGeGLUKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("geglu", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType,
			storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)

	if !ok {
		testingObject.Fatalf("geglu Metal kernel not registered for %s", storageDType.Name())
	}

	return kernel
}
