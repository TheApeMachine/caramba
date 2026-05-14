package compute

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/manifest"
)

func TestOptimizerContract(t *testing.T) {
	Convey("Given the standard optimizer backend contract", t, func() {
		Convey("It should match the train optimizer operation IDs", func() {
			contractIDs := StandardOptimizerContract.OperationIDs()
			operationIDs := ir.TrainOptimizerOperationIDs()

			So(contractIDs, ShouldResemble, operationIDs)
		})

		Convey("It should match manifest train optimizer registrations", func() {
			for _, operationID := range StandardOptimizerContract.OperationIDs() {
				operation, err := manifest.Build(string(operationID), map[string]any{})

				So(err, ShouldBeNil)
				So(operation, ShouldNotBeNil)
			}
		})
	})
}
