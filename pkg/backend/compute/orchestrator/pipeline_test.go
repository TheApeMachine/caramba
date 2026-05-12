package orchestrator

import (
	"context"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type recordingPass struct {
	name string
}

func (pass *recordingPass) Name() string {
	return pass.name
}

func (pass *recordingPass) Run(
	ctx context.Context,
	input PassInput,
) (PassResult, error) {
	_ = ctx
	input.Diagnostics.Add(pass.name, DiagnosticInfo, "ran")

	return PassResult{
		Graph:       input.Graph,
		Targets:     input.Targets,
		TargetMap:   input.TargetMap,
		Diagnostics: input.Diagnostics,
		Changed:     true,
	}, nil
}

func TestPipeline(t *testing.T) {
	Convey("Given a compiler pass pipeline", t, func() {
		shape, err := tensor.NewShape([]int{1})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		input := ir.NewNode("input", ir.OpInput, shape)
		graph.AddNode(input)

		pipeline := NewPipeline(&recordingPass{name: "first"}, &recordingPass{name: "second"})

		Convey("It should execute passes in order and preserve diagnostics", func() {
			result, err := pipeline.Run(context.Background(), graph, []*ir.Node{input})

			So(err, ShouldBeNil)
			So(result.Graph, ShouldEqual, graph)
			So(result.Targets, ShouldHaveLength, 1)
			So(result.Diagnostics.Messages(), ShouldHaveLength, 2)
			So(result.Diagnostics.Messages()[0].Pass, ShouldEqual, "first")
			So(result.Diagnostics.Messages()[1].Pass, ShouldEqual, "second")
		})
	})
}

func TestVerifierPass(t *testing.T) {
	Convey("Given a verifier pass", t, func() {
		shape, err := tensor.NewShape([]int{1})
		So(err, ShouldBeNil)

		graph := ir.NewGraph()
		target := ir.NewNode("target", ir.OpReLU, shape)
		target.AddInput(ir.NewNode("missing", ir.OpInput, shape))
		graph.AddNode(target)

		Convey("It should reject invalid graph structure before optimization", func() {
			_, err := NewPipeline(NewVerifierPass()).Run(
				context.Background(),
				graph,
				[]*ir.Node{target},
			)

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unregistered input")
		})
	})
}
