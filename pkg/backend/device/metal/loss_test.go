package metal

import (
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var metalPairLossNames = []string{
	"mse_loss",
	"mae_loss",
	"huber_loss",
	"binary_cross_entropy",
	"kl_divergence",
}

func TestKernelRegistry_MetalLossDTypes(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalLossDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			testMetalLossDType(testingObject, backend, storageDType)
		})
	}
}

func testMetalLossDType(
	testingObject *testing.T,
	backend *Backend,
	storageDType dtype.DType,
) {
	for _, elementCount := range parityElementCounts {
		elementCount := elementCount

		testingObject.Run(testNameForElementCount(elementCount), func(testingObject *testing.T) {
			convey.Convey("Given Metal "+storageDType.Name()+" tensors for losses", testingObject, func() {
				for _, name := range metalPairLossNames {
					runPairLossParityCase(testingObject, backend, name, storageDType, elementCount)
				}

				runCrossEntropyLossParityCase(testingObject, backend, storageDType, elementCount)
			})
		})
	}
}

func runPairLossParityCase(
	testingObject testing.TB,
	backend *Backend,
	name string,
	storageDType dtype.DType,
	elementCount int,
) {
	fixture := lossPairFixtureForTest(name, elementCount, storageDType)
	shape := mustShapeForTest(testingObject, []int{elementCount})
	outShape := mustShapeForTest(testingObject, []int{1})
	predictions := uploadDTypeTensorForTest(
		testingObject, backend, shape, storageDType, fixture.predictionBytes,
	)
	targets := uploadDTypeTensorForTest(testingObject, backend, shape, storageDType, fixture.targetBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(predictions, targets, out)

	err := lookupPairLossKernel(testingObject, name, storageDType).Run(predictions, targets, out)
	convey.So(err, convey.ShouldBeNil)
	assertLossBytesForTest(testingObject, backend, out, storageDType, fixture, lossMaxULP(name, storageDType))
}

func runCrossEntropyLossParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	classes int,
) {
	fixture := lossCrossEntropyFixtureForTest(classes, storageDType)
	batch := lossCrossEntropyBatch(classes)
	logitShape := mustShapeForTest(testingObject, []int{batch, classes})
	targetShape := mustShapeForTest(testingObject, []int{batch})
	outShape := mustShapeForTest(testingObject, []int{1})
	logits := uploadDTypeTensorForTest(testingObject, backend, logitShape, storageDType, fixture.logitBytes)
	targets := uploadDTypeTensorForTest(testingObject, backend, targetShape, dtype.Int32, fixture.targetBytes)
	out := emptyTensorForTest(testingObject, backend, outShape, storageDType)
	defer closeBenchmarkTensors(logits, targets, out)

	err := lookupCrossEntropyLossKernel(testingObject, storageDType).Run(logits, targets, out)
	convey.So(err, convey.ShouldBeNil)
	assertLossBytesForTest(
		testingObject,
		backend,
		out,
		storageDType,
		lossPairFixture{
			expectedBytes:   fixture.expectedBytes,
			expectedFloat32: fixture.expectedFloat32,
		},
		lossMaxULP("cross_entropy", storageDType),
	)
}

func lookupPairLossKernel(
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

func lookupCrossEntropyLossKernel(
	testingObject testing.TB,
	storageDType dtype.DType,
) kernels.Kernel {
	testingObject.Helper()

	kernel, ok := kernels.Default.LookupLocation("cross_entropy", kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  []dtype.DType{storageDType, dtype.Int32},
		Outputs: []dtype.DType{storageDType},
	}, tensor.Metal)
	if !ok {
		testingObject.Fatalf("missing Metal %s cross_entropy kernel", storageDType.Name())
	}

	return kernel
}

func assertLossBytesForTest(
	testingObject testing.TB,
	backend *Backend,
	input tensor.Tensor,
	storageDType dtype.DType,
	fixture lossPairFixture,
	maxULP uint32,
) {
	testingObject.Helper()

	if storageDType != dtype.Float32 {
		assertDTypeBytesForTest(testingObject, backend, input, storageDType, fixture.expectedBytes, maxULP)
		return
	}

	assertFloat32TensorForTest(testingObject, backend, input, fixture.expectedFloat32, maxULP)
}

func lossMaxULP(name string, storageDType dtype.DType) uint32 {
	if storageDType != dtype.Float32 {
		return 2
	}

	switch name {
	case "binary_cross_entropy", "kl_divergence", "cross_entropy":
		return 128
	default:
		return 4
	}
}
