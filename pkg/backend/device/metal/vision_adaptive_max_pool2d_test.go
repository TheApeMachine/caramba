package metal

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
)

func TestKernelRegistry_MetalAdaptiveMaxPool2D(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalVisionDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			for _, outputWidth := range parityElementCounts {
				outputWidth := outputWidth

				testingObject.Run(fmt.Sprintf("N=%d", outputWidth), func(testingObject *testing.T) {
					convey.Convey(
						"Given Metal "+storageDType.Name()+" adaptive_max_pool2d tensors",
						testingObject,
						func() {
							runAdaptiveMaxPool2DParityCase(
								testingObject, backend, storageDType, outputWidth,
							)
						},
					)
				})
			}
		})
	}
}

func TestKernelRegistry_MetalAdaptiveMaxPool2DSliceRegression(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range []dtype.DType{dtype.Float32, dtype.Float16, dtype.BFloat16} {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			convey.Convey(
				"Given Metal "+storageDType.Name()+" adaptive_avg_pool2d regression after adaptive_max_pool2d",
				testingObject,
				func() {
					runAdaptiveAvgPool2DParityCase(testingObject, backend, storageDType, 64)
				},
			)
		})
	}
}

func runAdaptiveMaxPool2DParityCase(
	testingObject testing.TB,
	backend *Backend,
	storageDType dtype.DType,
	outputWidth int,
) {
	inputBytes, expectedBytes := adaptiveMaxPool2DDTypeBytes(outputWidth, storageDType)
	input, out := adaptiveMaxPool2DTensorsForTest(
		testingObject, backend, outputWidth, storageDType, inputBytes,
	)
	defer closeBenchmarkTensors(input, out)

	err := lookupVisionPool2DKernel(testingObject, "adaptive_max_pool2d", storageDType).Run(input, out)
	convey.So(err, convey.ShouldBeNil)
	assertProjectionBytesForTest(testingObject, backend, out, storageDType, expectedBytes)
}
