package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalReductionNames = []string{
	"sum",
	"mean",
	"prod",
	"reduce_min",
	"reduce_max",
	"argmin",
	"argmax",
	"l1_norm",
	"l2_norm",
	"variance",
	"stddev",
}

func TestKernelRegistry_MetalReductionDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalReductionDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalReductionDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalReductionDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" tensors for reductions", testingObject, func() {
				for _, name := range metalReductionNames {
					runReductionParityCase(testingObject, backend, name, storageDType, elementCount)
				}
			})
		})
	}
}

func runReductionParityCase(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := reductionFixtureForTest(name, elementCount, storageDType)
	inputShape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{1})
	input := uploadDTypeTensorForTest(testingObject, backend, inputShape, storageDType, fixture.inputBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(input, out)

	err := lookupReductionKernel(testingObject, name, storageDType).Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertReductionBytesForTest(
		testingObject, backend, out, storageDType, fixture, reductionMaxULP(name, storageDType),
	)
}

func lookupReductionKernel(
	testingObject testing.TB,
	name string,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation(name, kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s %s kernel", storageDType.Name(), name)
	}

	return kernel
}

func assertReductionBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	fixture reductionFixture,
	maxULP uint32,
) {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, input, storageDType, fixture.expectedBytes, maxULP)
		return
	}

	assertFloat32TensorForTest(testingObject, backend, input, fixture.expectedFloat32, maxULP)
}

func reductionMaxULP(name string, storageDType dtype.DType) uint32 {
	if storageDType != dtype.Float32 {
		return 2
	}

	switch name {
	case "l2_norm", "stddev":
		return 8
	default:
		return 4
	}
}
