package orchestrator

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestBackendCapabilityParity(test *testing.T) {
	Convey("Given backend capability declarations", test, func() {
		Convey("It should declare every required operation for host dispatch", func() {
			capabilities := CapabilitiesForLocation(tensor.Host)
			operationIDs := ir.RequiredOperationIDs()

			So(operationIDs, ShouldNotBeEmpty)

			for _, operationID := range operationIDs {
				So(capabilities.Supports(operationID), ShouldBeTrue)
			}
		})

		Convey("It should keep accelerator capabilities tied to resident kernels", func() {
			cudaCapabilities := CapabilitiesForLocation(tensor.CUDA)
			metalCapabilities := CapabilitiesForLocation(tensor.Metal)
			xlaCapabilities := CapabilitiesForLocation(tensor.XLA)

			So(cudaCapabilities.Supports(ir.OpMatmul), ShouldBeTrue)
			So(cudaCapabilities.Supports("attention.sdpa"), ShouldBeFalse)
			So(metalCapabilities.Supports("attention.sdpa"), ShouldBeTrue)
			So(metalCapabilities.Supports("convolution.conv2d"), ShouldBeTrue)
			So(metalCapabilities.Supports("convolution.conv1d"), ShouldBeFalse)
			So(xlaCapabilities.Supports(ir.OpGELU), ShouldBeTrue)
			So(xlaCapabilities.Supports("projection.linear"), ShouldBeFalse)
		})

		Convey("It should not use wildcard support as a substitute for explicit coverage", func() {
			for _, location := range []tensor.Location{
				tensor.Host,
				tensor.CUDA,
				tensor.Metal,
				tensor.XLA,
			} {
				So(CapabilitiesForLocation(location).Supports("*"), ShouldBeFalse)
			}
		})
	})
}

func BenchmarkCapabilitiesForLocation(benchmark *testing.B) {
	for benchmark.Loop() {
		capabilities := CapabilitiesForLocation(tensor.Metal)
		_ = capabilities.Supports("attention.sdpa")
	}
}
