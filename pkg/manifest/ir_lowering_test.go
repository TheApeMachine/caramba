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

	Convey("Given a manifest graph with declared external inputs", t, func() {
		graph := newGraph()
		graph.externalInputs["x"] = true
		So(graph.addNode(&Node{
			ID:     "activation",
			OpID:   "activation.relu",
			Config: map[string]any{},
			In:     []string{"x"},
			Out:    []string{"y"},
		}), ShouldBeNil)
		err := graph.rebuildEdgesFromNodes()
		So(err, ShouldBeNil)

		shape, err := tensor.NewShape([]int{4})
		So(err, ShouldBeNil)

		Convey("It should materialize external inputs as IR input nodes", func() {
			irGraph, err := LowerGraphToIR(graph, shape)

			So(err, ShouldBeNil)

			index, err := irGraph.Index()
			So(err, ShouldBeNil)

			input := index.Node("x")
			So(input, ShouldNotBeNil)
			So(input.OpType(), ShouldEqual, ir.OpInput)
			So(input.OperationID(), ShouldEqual, ir.OpID("data.input"))
			So(index.Node("activation").Inputs()[0].ID(), ShouldEqual, "x")
		})
	})

	Convey("Given a transformer-shaped manifest graph", t, func() {
		graph := newGraph()
		graph.externalInputs["input_ids"] = true
		So(graph.addNode(&Node{
			ID:     "token_embedding",
			OpID:   "embedding.token",
			Config: map[string]any{"d_model": 4, "vocab_size": 8},
			In:     []string{"input_ids"},
			Out:    []string{"hidden"},
		}), ShouldBeNil)
		So(graph.addNode(&Node{
			ID:     "projection",
			OpID:   "projection.linear",
			Config: map[string]any{"in_features": 4, "out_features": 6},
			In:     []string{"hidden"},
			Out:    []string{"logits"},
		}), ShouldBeNil)
		So(graph.addNode(&Node{
			ID:     "last_token",
			OpID:   "shape.last_token",
			Config: map[string]any{},
			In:     []string{"hidden"},
			Out:    []string{"last_hidden"},
		}), ShouldBeNil)
		So(graph.rebuildEdgesFromNodes(), ShouldBeNil)

		shape, err := tensor.NewShape([]int{1, 3})
		So(err, ShouldBeNil)

		Convey("It should infer output shapes while preserving operation shapes", func() {
			irGraph, err := LowerGraphToIR(graph, shape)

			So(err, ShouldBeNil)

			index, err := irGraph.Index()
			So(err, ShouldBeNil)

			embedding := index.Node("token_embedding")
			projection := index.Node("projection")
			lastToken := index.Node("last_token")

			So(embedding.Shape().Dims(), ShouldResemble, []int{1, 3, 4})
			So(embedding.Metadata()["op_shape"], ShouldResemble, []int{1, 3})
			So(projection.Shape().Dims(), ShouldResemble, []int{1, 3, 6})
			So(projection.Metadata()["op_shape"], ShouldResemble, []int{1, 3, 4})
			So(lastToken.Shape().Dims(), ShouldResemble, []int{1, 1, 4})
			So(lastToken.Metadata()["op_shape"], ShouldResemble, []int{1, 3, 4})
		})
	})
}
