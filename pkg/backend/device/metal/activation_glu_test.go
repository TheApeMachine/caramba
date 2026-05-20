package metal

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

const gluMaxULP uint32 = 2

func TestKernelRegistry_MetalGLUDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalGLUDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			for _, elementCount := range parityElementCounts {
				elementCount := elementCount

				testingObject.Run(fmt.Sprintf("N=%d", elementCount), func(testingObject *testing.T) {
					convey.Convey(
						"Given Metal "+storageDType.Name()+" tensors for glu",
						testingObject,
						func() {
							runGLUParityCase(testingObject, backend, storageDType, elementCount)
						},
					)
				})
			}
		})
	}
}

func runGLUParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := gluFixtureForTest(elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	gate := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.gateBytes)
	up := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.upBytes)
	out := emptyTensorForTest(testingObject, backend, shape, storageDType)
	defer closeBenchmarkTensors(gate, up, out)

	err := lookupGLUKernel(testingObject, storageDType).Run(gate, up, out)
	convey.So(err, convey.ShouldBeNil)

	if storageDType == dtype.Float32 {
		actualDType, actualBytes, downloadErr := backend.Download(out)
		convey.So(downloadErr, convey.ShouldBeNil)
		convey.So(actualDType, convey.ShouldEqual, dtype.Float32)
		assertFloat32WithinULP(
			testingObject,
			mustFloat32Bytes(actualBytes),
			mustFloat32Bytes(fixture.expectedBytes),
			gluMaxULP,
		)
		return
	}

	assertDTypeBytesForTest(testingObject, backend, out, storageDType, fixture.expectedBytes, gluMaxULP)
}

func lookupGLUKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("glu", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType,
			storageDType,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)

	if !ok {
		testingObject.Fatalf("glu Metal kernel not registered for %s", storageDType.Name())
	}

	return kernel
}
