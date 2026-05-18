package backend

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/backend/compute"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

const reluManifest = `
system:
  topology:
    inputs: [x]
    nodes:
      - id: relu
        op: activation.relu
        in: [x]
        out: [y]
`

const unusedInputManifest = `
system:
  topology:
    inputs: [x, timestep]
    nodes:
      - id: relu
        op: activation.relu
        in: [x]
        out: [y]
`

func writeManifest(t *testing.T, root, relative, body string) {
	t.Helper()
	full := filepath.Join(root, relative)
	So(os.MkdirAll(filepath.Dir(full), 0o755), ShouldBeNil)
	So(os.WriteFile(full, []byte(body), 0o644), ShouldBeNil)
}

func TestGraphRunnerCall(t *testing.T) {
	Convey("Given a CPU compute backend and a tiny ReLU topology manifest", t, func() {
		root := t.TempDir()
		writeManifest(t, root, "relu.yml", reluManifest)

		backend, err := compute.NewBackend(compute.CPU)
		So(err, ShouldBeNil)
		defer backend.Close()

		runner, err := New(Options{
			ComputeBackend: backend,
			ProjectRoot:    root,
		})
		So(err, ShouldBeNil)

		module := program.GraphModule{
			ID:       "relu",
			Manifest: "relu.yml",
			Config: map[string]any{
				"outputs": map[string]any{"y": "relu"},
			},
		}

		Convey("Calling the runner should apply ReLU element-wise", func() {
			outputs, err := runner.Call(context.Background(), module, map[string]any{
				"x": []float64{-1, 0, 2, -3, 4},
			})

			So(err, ShouldBeNil)
			values, ok := outputs["y"].([]float64)
			So(ok, ShouldBeTrue)
			So(values, ShouldResemble, []float64{0, 0, 2, 0, 4})
		})

		Convey("Calling the runner without the manifest path should error clearly", func() {
			noManifestModule := program.GraphModule{
				ID:       "relu",
				Manifest: "",
			}

			_, err := runner.Call(context.Background(), noManifestModule, map[string]any{})
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "no manifest path declared")
		})
	})

	Convey("Given a graph call binds an unused topology input", t, func() {
		root := t.TempDir()
		writeManifest(t, root, "unused.yml", unusedInputManifest)

		backend, err := compute.NewBackend(compute.CPU)
		So(err, ShouldBeNil)
		defer backend.Close()

		runner, err := New(Options{
			ComputeBackend: backend,
			ProjectRoot:    root,
		})
		So(err, ShouldBeNil)

		module := program.GraphModule{
			ID:       "unused",
			Manifest: "unused.yml",
			Config: map[string]any{
				"outputs": map[string]any{"y": "relu"},
			},
		}

		Convey("Call should reject the graph before execution", func() {
			_, err := runner.Call(context.Background(), module, map[string]any{
				"x":        []float64{-1, 2},
				"timestep": 999.0,
			})

			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, `input "timestep" is declared by the graph but has no consumers`)
		})
	})

	Convey("Given the runner is constructed without a compute backend", t, func() {
		_, err := New(Options{ComputeBackend: nil})
		So(err, ShouldNotBeNil)
		So(err.Error(), ShouldContainSubstring, "compute backend is required")
	})
}
