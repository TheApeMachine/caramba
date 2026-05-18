package manifest

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
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
			So(irGraph.Nodes()[1].ValueType().Precision, ShouldEqual, dtype.Float32)
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

	Convey("Given a logsumexp-shaped manifest graph", t, func() {
		graph := newGraph()
		graph.externalInputs["energy"] = true
		So(graph.addNode(&Node{
			ID:     "log_partition",
			OpID:   "math.logsumexp",
			Config: map[string]any{},
			In:     []string{"energy"},
			Out:    []string{"z"},
		}), ShouldBeNil)
		So(graph.rebuildEdgesFromNodes(), ShouldBeNil)

		shape, err := tensor.NewShape([]int{2, 3, 5})
		So(err, ShouldBeNil)

		Convey("It should reduce the last dimension", func() {
			irGraph, err := LowerGraphToIR(graph, shape)

			So(err, ShouldBeNil)

			index, err := irGraph.Index()
			So(err, ShouldBeNil)

			node := index.Node("log_partition")

			So(node.Shape().Dims(), ShouldResemble, []int{2, 3})
			So(node.Metadata()["op_shape"], ShouldResemble, []int{2, 3, 5})
		})
	})

	Convey("Given a manifest graph with multiple input shapes", t, func() {
		graph := newGraph()
		graph.externalInputs["left"] = true
		graph.externalInputs["right"] = true
		So(graph.addNode(&Node{
			ID:     "concat",
			OpID:   "shape.concat",
			Config: map[string]any{"dim": 2},
			In:     []string{"left", "right"},
			Out:    []string{"joined"},
		}), ShouldBeNil)
		So(graph.rebuildEdgesFromNodes(), ShouldBeNil)

		defaultShape, err := tensor.NewShape([]int{1, 4, 3})
		So(err, ShouldBeNil)
		rightShape, err := tensor.NewShape([]int{1, 4, 5})
		So(err, ShouldBeNil)

		Convey("It should infer concat output from each input shape", func() {
			irGraph, err := LowerGraphToIRWithInputShapes(
				graph,
				defaultShape,
				map[string]tensor.Shape{"right": rightShape},
			)
			So(err, ShouldBeNil)

			index, err := irGraph.Index()
			So(err, ShouldBeNil)

			So(index.Node("concat").Shape().Dims(), ShouldResemble, []int{1, 4, 8})
			So(index.Node("right").Shape().Dims(), ShouldResemble, []int{1, 4, 5})
		})
	})

	Convey("Given a shape.slice node with a negative end", t, func() {
		graph := newGraph()
		graph.externalInputs["x"] = true
		So(graph.addNode(&Node{
			ID:     "slice",
			OpID:   "shape.slice",
			Config: map[string]any{"dim": 1, "start": 0, "end": -1},
			In:     []string{"x"},
			Out:    []string{"y"},
		}), ShouldBeNil)
		So(graph.rebuildEdgesFromNodes(), ShouldBeNil)

		shape, err := tensor.NewShape([]int{2, 8})
		So(err, ShouldBeNil)

		Convey("It should reject the end value with a targeted error", func() {
			_, err := LowerGraphToIR(graph, shape)

			So(err, ShouldNotBeNil)
			So(
				err.Error(),
				ShouldContainSubstring,
				"manifest: shape.slice end -1 out of range for dim 1 size 8",
			)
		})
	})

	Convey("Given a convolution-shaped manifest graph", t, func() {
		graph := newGraph()
		graph.externalInputs["image"] = true
		So(graph.addNode(&Node{
			ID:   "conv",
			OpID: "convolution.conv2d",
			Config: map[string]any{
				"out_channels": 8,
				"kernel_h":     3,
				"kernel_w":     3,
				"stride_h":     2,
				"stride_w":     2,
				"pad_h":        1,
				"pad_w":        1,
			},
			In:  []string{"image"},
			Out: []string{"hidden"},
		}), ShouldBeNil)
		So(graph.addNode(&Node{
			ID:   "nearest",
			OpID: "shape.upsample_nearest2d",
			Config: map[string]any{
				"scale_h": 2,
				"scale_w": 2,
			},
			In:  []string{"hidden"},
			Out: []string{"nearest"},
		}), ShouldBeNil)
		So(graph.addNode(&Node{
			ID:   "up",
			OpID: "convolution.conv_transpose2d",
			Config: map[string]any{
				"out_channels": 4,
				"kernel_h":     4,
				"kernel_w":     4,
				"stride_h":     2,
				"stride_w":     2,
				"pad_h":        1,
				"pad_w":        1,
			},
			In:  []string{"nearest"},
			Out: []string{"up"},
		}), ShouldBeNil)
		So(graph.rebuildEdgesFromNodes(), ShouldBeNil)

		shape, err := tensor.NewShape([]int{1, 3, 32, 32})
		So(err, ShouldBeNil)

		Convey("It should infer convolution output shapes", func() {
			irGraph, err := LowerGraphToIR(graph, shape)

			So(err, ShouldBeNil)

			index, err := irGraph.Index()
			So(err, ShouldBeNil)

			So(index.Node("conv").Shape().Dims(), ShouldResemble, []int{1, 8, 16, 16})
			So(index.Node("nearest").Shape().Dims(), ShouldResemble, []int{1, 8, 32, 32})
			So(index.Node("up").Shape().Dims(), ShouldResemble, []int{1, 4, 64, 64})
		})
	})
}
