package metal

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestKernelRegistry_MetalConvTranspose2D(testingObject *testing.T) {
	backend := newBackendForDeviceTest(testingObject)
	defer func() {
		if err := backend.Close(); err != nil {
			testingObject.Fatalf("Close failed: %v", err)
		}
	}()

	for _, storageDType := range metalVisionDTypes {
		storageDType := storageDType

		testingObject.Run(storageDType.Name(), func(testingObject *testing.T) {
			for _, width := range parityElementCounts {
				width := width

				testingObject.Run(fmt.Sprintf("N=%d", width), func(testingObject *testing.T) {
					convey.Convey(
						"Given Metal "+storageDType.Name()+" conv_transpose2d tensors",
						testingObject,
						func() {
							runConvTranspose2DParityCase(testingObject, backend, storageDType, width)
						},
					)
				})
			}
		})
	}
}

func TestKernelRegistry_MetalConvTranspose2DSliceRegression(testingObject *testing.T) {
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
				"Given Metal "+storageDType.Name()+" slice regression after conv_transpose2d",
				testingObject,
				func() {
					runMetalSliceShapeCase(testingObject, backend, storageDType, 64)
				},
			)
		})
	}
}
