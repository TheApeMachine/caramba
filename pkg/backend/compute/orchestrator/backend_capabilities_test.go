package orchestrator

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestBackendCapabilityParity(t *testing.T) {
	Convey("Given full backend capability declarations", t, func() {
		for _, location := range []tensor.Location{
			tensor.Host,
			tensor.CUDA,
			tensor.Metal,
			tensor.XLA,
		} {
			location := location

			Convey("It should declare every required operation for "+string(location), func() {
				capabilities := CapabilitiesForLocation(location)

				for _, operationID := range ir.RequiredOperationIDs() {
					So(capabilities.Supports(operationID), ShouldBeTrue)
				}
			})
		}

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
