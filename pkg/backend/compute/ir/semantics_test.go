package ir

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func TestNodeSemantics(t *testing.T) {
	Convey("Given an IR node with typed compiler semantics", t, func() {
		shape, err := tensor.NewShape([]int{2, 3})
		So(err, ShouldBeNil)

		node := NewNode("projection", OpMatmul, shape)
		node.SetOperationID("math.matmul")
		node.SetValueType(ValueType{
			DType:       tensor.Float64,
			Shape:       shape,
			Layout:      LayoutRowMajor,
			MemoryClass: MemoryDevice,
		})
		node.SetEffect(EffectPure)
		node.SetAlias(Alias{
			Kind:       AliasAllocates,
			InPlace:    false,
			InputIndex: -1,
		})
		node.SetAttribute("beta", FloatAttribute(0.5))
		node.SetAttribute("alpha", IntAttribute(1))

		Convey("It should expose deterministic operation, type, effect, alias, and attributes", func() {
			So(node.OperationID(), ShouldEqual, OpID("math.matmul"))
			So(node.ValueType().DType, ShouldEqual, tensor.Float64)
			So(node.ValueType().Layout, ShouldEqual, LayoutRowMajor)
			So(node.ValueType().MemoryClass, ShouldEqual, MemoryDevice)
			So(node.Effect(), ShouldEqual, EffectPure)
			So(node.Alias().Kind, ShouldEqual, AliasAllocates)
			So(node.CanonicalAttributes(), ShouldEqual, "alpha=i:1;beta=f:0.5;")
		})
	})
}

func TestGraphVerify(t *testing.T) {
	Convey("Given an IR graph verifier", t, func() {
		shape, err := tensor.NewShape([]int{1})
		So(err, ShouldBeNil)

		Convey("It should reject inputs that are not registered in the graph", func() {
			graph := NewGraph()
			input := NewNode("input", OpInput, shape)
			output := NewNode("output", OpReLU, shape)
			output.AddInput(input)
			graph.AddNode(output)

			err := graph.Verify()

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unregistered input")
		})

		Convey("It should build indexes and clone graphs while preserving target remaps", func() {
			graph := NewGraph()
			input := NewNode("input", OpInput, shape)
			output := NewNode("output", OpReLU, shape)
			output.AddInput(input)
			graph.AddNode(input)
			graph.AddNode(output)

			index, err := graph.Index()
			So(err, ShouldBeNil)
			So(index.Node("input"), ShouldEqual, input)
			So(index.Users("input"), ShouldHaveLength, 1)

			clone, replacements, err := graph.Clone()
			So(err, ShouldBeNil)
			So(clone, ShouldNotBeNil)
			So(replacements["output"] == output, ShouldBeFalse)
			So(replacements["output"].Inputs()[0].ID(), ShouldEqual, "input")
		})
	})
}
