package integration

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/backend/compute"
	"github.com/theapemachine/caramba/pkg/runtime/backend"
	"github.com/theapemachine/caramba/pkg/runtime/compiler"
	"github.com/theapemachine/caramba/pkg/runtime/executor"
	_ "github.com/theapemachine/caramba/pkg/runtime/op/builtins"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

const reluTopologyYAML = `
system:
  topology:
    inputs: [x]
    nodes:
      - id: relu
        op: activation.relu
        in: [x]
        out: [y]
`

const reluRuntimeYAML = `
name: relu_runtime
system:
  runtime:
    type: program
    state:
      input_tensor:
        type: tensor
        shape: [5]
      output_tensor:
        type: tensor
        shape: [5]
    graphs:
      relu:
        manifest: relu.yml
        outputs:
          y: relu
    program:
      - id: forward
        op: graph.call
        graph: relu
        inputs:
          x: state.input_tensor
        outputs:
          y: local_y
`

func TestRuntimeOverRealBackend(t *testing.T) {
	Convey("Given a YAML runtime program that calls a real ReLU topology", t, func() {
		root := t.TempDir()
		So(os.WriteFile(filepath.Join(root, "relu.yml"), []byte(reluTopologyYAML), 0o644), ShouldBeNil)

		runtimeProgram, err := compiler.New(root).CompileBytes([]byte(reluRuntimeYAML))
		So(err, ShouldBeNil)

		computeBackend, err := compute.NewBackend(compute.CPU)
		So(err, ShouldBeNil)
		defer computeBackend.Close()

		graphRunner, err := backend.New(backend.Options{
			ComputeBackend: computeBackend,
			ProjectRoot:    root,
		})
		So(err, ShouldBeNil)

		exec, err := executor.New(executor.Options{
			Program:     runtimeProgram,
			GraphRunner: graphRunner,
		})
		So(err, ShouldBeNil)

		Convey("Seeding the input tensor and running should apply ReLU", func() {
			inputTensor := exec.States()["input_tensor"].(*state.Tensor)
			So(inputTensor.Set([]int{5}, []float64{-2, -1, 0, 1, 2}), ShouldBeNil)

			So(exec.Run(context.Background()), ShouldBeNil)
		})
	})
}
