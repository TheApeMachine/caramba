package ir

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestGraphCodec(t *testing.T) {
	Convey("Given a typed IR graph", t, func() {
		shape, err := tensor.NewShape([]int{2})
		So(err, ShouldBeNil)

		graph := NewGraph()
		input := NewNode("input", OpInput, shape)
		input.SetOperationID("data.input")
		output := NewNode("output", OpReLU, shape)
		output.SetOperationID("activation.relu")
		output.SetAttribute("alpha", FloatAttribute(0.1))
		output.AddInput(input)
		graph.AddNode(input)
		graph.AddNode(output)

		Convey("It should encode and decode a versioned representation", func() {
			data, err := EncodeGraph(graph)
			So(err, ShouldBeNil)
			So(string(data), ShouldContainSubstring, `"version":1`)

			decoded, err := DecodeGraph(data)
			So(err, ShouldBeNil)
			So(decoded.Nodes(), ShouldHaveLength, 2)
			So(decoded.Nodes()[1].OperationID(), ShouldEqual, OpID("activation.relu"))
			So(decoded.Nodes()[1].Inputs()[0].ID(), ShouldEqual, "input")
		})
	})
}
