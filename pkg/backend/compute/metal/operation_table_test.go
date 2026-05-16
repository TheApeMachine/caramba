package metal

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestResidentOperationTable(test *testing.T) {
	Convey("Given the Metal resident operation table", test, func() {
		table := ResidentOperationTable()

		Convey("It should declare concrete resident coverage metadata", func() {
			So(table, ShouldNotBeEmpty)

			seen := make(map[ir.OpType]bool, len(table))

			for _, operation := range table {
				So(string(operation.ID), ShouldNotBeBlank)
				So(operation.ResidentSymbol, ShouldNotBeBlank)
				So(operation.BenchmarkName, ShouldNotBeBlank)
				So(operation.ParityTestName, ShouldNotBeBlank)
				So(operation.DTypes, ShouldResemble, []tensor.DType{tensor.Float32})
				So(seen[operation.ID], ShouldBeFalse)

				seen[operation.ID] = true
			}
		})

		Convey("It should expose lookup copies instead of mutable table entries", func() {
			operation, ok := ResidentOperationByID("attention.sdpa")

			So(ok, ShouldBeTrue)
			So(operation.ResidentSymbol, ShouldEqual, "metal_sdpa_tensor")

			operation.DTypes[0] = tensor.Float64
			second, ok := ResidentOperationByID("attention.sdpa")

			So(ok, ShouldBeTrue)
			So(second.DTypes, ShouldResemble, []tensor.DType{tensor.Float32})
		})
	})
}

func BenchmarkResidentOperationTable(benchmark *testing.B) {
	for benchmark.Loop() {
		_ = ResidentOperationTable()
	}
}
