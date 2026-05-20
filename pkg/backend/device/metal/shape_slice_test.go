package metal

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestKernelRegistry_MetalSliceDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalShapeDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			for _, elementCount := range parityElementCounts {
				elementCount := elementCount

				testingObject.Run(fmt.Sprintf("N=%d", elementCount), func(testingObject *testing.T) {
					convey.Convey(
						"Given Metal "+storageDType.Name()+" tensors for slice",
						testingObject,
						func() {
							runMetalSliceShapeCase(testingObject, backend, storageDType, elementCount)
						},
					)
				})
			}
		})
	}
}

func runMetalSliceShapeCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	seqLen int,
) {
	fixture := sliceFixtureForTest(seqLen, storageDType)
	inputShape := mustShapeForTest(testingObject, sliceInputShapeForTest(seqLen))
	outShape := mustShapeForTest(testingObject, sliceOutputShapeForTest(seqLen))
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, fixture.inputBytes)
	dim := uploadInt32ScalarForTest(testingObject, backend, 1)
	start := uploadInt32ScalarForTest(testingObject, backend, 0)
	end := uploadInt32ScalarForTest(testingObject, backend, int32(seqLen))
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(input, dim, start, end, out)

	err := lookupSliceKernel(testingObject, storageDType).Run(input, dim, start, end, out)
	convey.So(err, convey.ShouldBeNil)
	assertRawBytesForTest(testingObject, backend, out, storageDType, fixture.expectedBytes)
}

func lookupSliceKernel(testingObject testing.TB, storageDType dtype.DType) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("slice", kernels.Signature{
		Layout: tensor.LayoutDense,
		Inputs: []dtype.DType{
			storageDType,
			dtype.Int32,
			dtype.Int32,
			dtype.Int32,
		},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s slice kernel", storageDType.Name())
	}

	return kernel
}
