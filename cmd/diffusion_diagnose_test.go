package cmd

import (
	"bytes"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/compiler"
	"github.com/theapemachine/manifesto/ir"
)

func TestWriteWorkspaceReport(testingObject *testing.T) {
	convey.Convey("Given compiled graph workspace layouts", testingObject, func() {
		output := &compiler.CompileOutput{
			Graphs: map[string]*ast.Graph{
				"denoiser": {
					Nodes: []*ast.GraphNode{
						{ID: "a"},
						{ID: "b"},
					},
				},
			},
			Workspaces: map[string]*ir.Topology{
				"denoiser": {
					Workspace: ir.WorkspaceLayout{
						Size: 2 * 1024 * 1024,
						Allocations: []ir.Interval{
							{PortID: 1, Size: 64},
							{PortID: 2, Size: 512 * 1024},
						},
					},
				},
			},
		}

		var buffer bytes.Buffer
		err := writeWorkspaceReport(&buffer, "runtime/diffusion.yml", output)

		convey.Convey("It should print graph memory pressure", func() {
			convey.So(err, convey.ShouldBeNil)
			convey.So(buffer.String(), convey.ShouldContainSubstring, "runtime/diffusion.yml")
			convey.So(buffer.String(), convey.ShouldContainSubstring, "denoiser")
			convey.So(buffer.String(), convey.ShouldContainSubstring, "2.00 MiB")
			convey.So(buffer.String(), convey.ShouldContainSubstring, "512.00 KiB")
		})
	})
}

func TestDefaultPlannerBindings(testingObject *testing.T) {
	convey.Convey("Given the runtime planner bindings", testingObject, func() {
		bindings := defaultPlannerBindings()

		convey.Convey("It should include diffusion and sequence bounds", func() {
			convey.So(bindings["N"], convey.ShouldEqual, int64(4096))
			convey.So(bindings["T"], convey.ShouldEqual, int64(4096))
			convey.So(bindings["B"], convey.ShouldEqual, int64(1))
		})
	})
}
