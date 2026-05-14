package cpu

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/operations"
)

func TestOperationDispatchParity(t *testing.T) {
	Convey("Given the CPU operation dispatcher", t, func() {
		Convey("It should expose every required catalog operation", func() {
			supported := TensorOperationDispatchContract.SupportedIDSet()

			for _, spec := range operations.Canonical.Required() {
				So(supported[spec.ID], ShouldBeTrue)
			}
		})
	})
}
