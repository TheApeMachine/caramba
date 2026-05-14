package manifest

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type scaleOp struct{ factor float64 }

func (scale *scaleOp) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("math.scale"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 || stateDict.Inputs[0] == nil {
		return nil, fmt.Errorf("math.scale: input[0] is required")
	}

	if stateDict.Out == nil {
		return nil, fmt.Errorf("math.scale: output is required")
	}

	if len(stateDict.Out) < len(stateDict.Inputs[0]) {
		return nil, fmt.Errorf("math.scale: output length %d < input length %d", len(stateDict.Out), len(stateDict.Inputs[0]))
	}

	for index, value := range stateDict.Inputs[0] {
		stateDict.Out[index] = value * scale.factor
	}

	return stateDict, nil
}

func TestCompiler_Compile(t *testing.T) {
	Convey("Given a Compiler with an isolated operation registry", t, func() {
		operationRegistry := NewOperationRegistry()

		operationRegistry.Register("math.scale", func(cfg map[string]any) (operation.Operation, error) {
			factor, _ := cfg["factor"].(float64)

			return &scaleOp{factor: factor}, nil
		})

		root := t.TempDir()

		write := func(rel, content string) {
			path := filepath.Join(root, rel)
			So(os.MkdirAll(filepath.Dir(path), 0o755), ShouldBeNil)
			So(os.WriteFile(path, []byte(content), 0o644), ShouldBeNil)
		}

		compiler := NewCompilerWithRegistry(root, operationRegistry)

		Convey("Compile", func() {
			Convey("It should build a graph from a valid manifest", func() {
				write("master.yml", `
system:
    topology:
      type: GraphTopology
      inputs: [x]
      nodes:
      - id: scale
        op: math.scale
        in: [x]
        out: [y]
        config:
          factor: 3.0
`)
				graph, err := compiler.Compile("master.yml")
				So(err, ShouldBeNil)
				So(graph, ShouldNotBeNil)
				So(graph.nodes, ShouldHaveLength, 1)
			})

			Convey("It should execute the graph correctly", func() {
				write("master.yml", `
system:
    topology:
      type: GraphTopology
      inputs: [x]
      nodes:
      - id: scale
        op: math.scale
        in: [x]
        out: [y]
        config:
          factor: 2.0
`)
				graph, err := compiler.Compile("master.yml")
				So(err, ShouldBeNil)

				result, err := graph.Execute(map[string][]float64{"x": {1, 2, 3, 4}}, []int{4})
				So(err, ShouldBeNil)
				So(result["y"], ShouldResemble, []float64{2, 4, 6, 8})
			})

			Convey("It should return an error for an unknown operation", func() {
				write("bad.yml", `
system:
  topology:
    inputs: [x]
    nodes:
      - id: mystery
        op: does.not.exist
        in: [x]
        out: [y]
`)
				_, err := compiler.Compile("bad.yml")
				So(err, ShouldNotBeNil)
			})

			Convey("It should return an error when system key is missing", func() {
				write("nosystem.yml", "name: bare\n")
				_, err := compiler.Compile("nosystem.yml")
				So(err, ShouldNotBeNil)
			})

			Convey("It should return an error when topology lacks nodes", func() {
				write("nonodes.yml", "system:\n  topology:\n    type: GraphTopology\n")
				_, err := compiler.Compile("nonodes.yml")
				So(err, ShouldNotBeNil)
			})

			Convey("It should reject undeclared external input bindings", func() {
				write("missing-input.yml", `
system:
  topology:
    nodes:
      - id: scale
        op: math.scale
        in: [x]
        out: [y]
`)
				_, err := compiler.Compile("missing-input.yml")
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "has no producer or declared external input")
			})

			Convey("It should reject duplicate node IDs", func() {
				write("duplicate.yml", `
system:
  topology:
    inputs: [x]
    nodes:
      - id: scale
        op: math.scale
        in: [x]
        out: [y]
      - id: scale
        op: math.scale
        in: [x]
        out: [z]
`)
				_, err := compiler.Compile("duplicate.yml")
				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "duplicate node id")
			})
		})
	})
}

func BenchmarkCompiler_Compile(b *testing.B) {
	operationRegistry := NewOperationRegistry()

	operationRegistry.Register("math.scale", func(cfg map[string]any) (operation.Operation, error) {
		return &scaleOp{factor: 1.0}, nil
	})

	root := b.TempDir()
	content := `
system:
  topology:
    inputs: [x]
    nodes:
      - id: s
        op: math.scale
        in: [x]
        out: [y]
        config:
          factor: 1.0
`

	manifestPath := filepath.Join(root, "bench.yml")

	_ = os.WriteFile(manifestPath, []byte(content), 0o644)

	compiler := NewCompilerWithRegistry(root, operationRegistry)

	b.ResetTimer()

	for b.Loop() {
		_, _ = compiler.Compile("bench.yml")
	}
}
