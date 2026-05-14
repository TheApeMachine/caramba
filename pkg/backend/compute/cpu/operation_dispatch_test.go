package cpu

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
)

func TestOperationDispatchParity(t *testing.T) {
	Convey("Given the CPU operation dispatcher", t, func() {
		Convey("It should expose every required catalog operation", func() {
			supported := TensorOperationDispatchContract.SupportedIDSet()

			for _, operationID := range ir.RequiredOperationIDs() {
				So(supported[operationID], ShouldBeTrue)
			}
		})
	})
}
