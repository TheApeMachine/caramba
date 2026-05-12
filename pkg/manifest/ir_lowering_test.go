package manifest

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestLowerGraphToIR(t *testing.T) {
	Convey("Given a manifest graph with operation IDs and port bindings", t, func() {
		graph := newGraph()
		So(graph.addNode(&Node{
			ID:     "input",
			OpID:   "data.input",
			Config: map[string]any{"dtype": "float64"},
			Out:    []string{"x"},
		}), ShouldBeNil)
		So(graph.addNode(&Node{
			ID:     "activation",
			OpID:   "activation.relu",
			Config: map[string]any{"inplace": false},
			In:     []string{"x"},
			Out:    []string{"y"},
		}), ShouldBeNil)
		err := graph.rebuildEdgesFromNodes()
		So(err, ShouldBeNil)

		shape, err := tensor.NewShape([]int{4})
		So(err, ShouldBeNil)

		Convey("It should preserve stable node IDs, op IDs, configs, and port bindings", func() {
			irGraph, err := LowerGraphToIR(graph, shape)

			So(err, ShouldBeNil)
			nodes := irGraph.Nodes()
			So(nodes, ShouldHaveLength, 2)
			So(nodes[0].OperationID(), ShouldEqual, ir.OpID("data.input"))
			So(nodes[1].OperationID(), ShouldEqual, ir.OpID("activation.relu"))
			So(nodes[1].Metadata()["inplace"], ShouldEqual, false)
			So(nodes[1].Attribute("in.0").String(), ShouldEqual, "s:x")
			So(nodes[1].Attribute("out.0").String(), ShouldEqual, "s:y")
			So(nodes[1].Inputs()[0].ID(), ShouldEqual, "input")
		})

		Convey("It should lower explicit precision opt-in", func() {
			graph.nodes[1].Config["precision"] = "float32"

			irGraph, err := LowerGraphToIR(graph, shape)

			So(err, ShouldBeNil)
			So(irGraph.Nodes()[1].ValueType().Precision, ShouldEqual, tensor.Float32)
		})
	})
}
