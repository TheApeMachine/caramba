package backend

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestApplyDefaultPrecision(t *testing.T) {
	Convey("Given an IR graph with mixed precision settings", t, func() {
		shape, err := tensor.NewShape([]int{4})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		input := ir.NewNode("input", ir.OpInput, shape)
		relu := ir.NewNode("relu", ir.OpReLU, shape)
		relu.SetValueType(ir.ValueType{
			Shape:     shape,
			DType:     dtype.Float64,
			Precision: dtype.Float64,
		})
		graph.AddNode(input)
		graph.AddNode(relu)

		Convey("Applying float32 default should rewrite every node's precision", func() {
			applyDefaultPrecision(graph, dtype.Float32)

			So(input.ValueType().Precision, ShouldEqual, dtype.Float32)
			So(relu.ValueType().Precision, ShouldEqual, dtype.Float32)
		})

		Convey("Empty default should be a no-op", func() {
			initialPrecision := input.ValueType().Precision

			applyDefaultPrecision(graph, dtype.Invalid)

			So(input.ValueType().Precision, ShouldEqual, initialPrecision)
		})
	})
}
