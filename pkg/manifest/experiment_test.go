package manifest

import (
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestCompiler_CompileExperiment(t *testing.T) {
	Convey("Given a Compiler anchored at a temp project root", t, func() {
		root := t.TempDir()

		write := func(rel, content string) {
			path := filepath.Join(root, rel)
			So(os.MkdirAll(filepath.Dir(path), 0o755), ShouldBeNil)
			So(os.WriteFile(path, []byte(content), 0o644), ShouldBeNil)
		}

		compiler := NewCompiler(root)

		Convey("CompileExperiment", func() {
			Convey("It should compile targets into named graphs", func() {
				write("exp.yml", `
variables:
  model:
    dim: 64

datasets:
  - name: train_set
    url: https://example.com/data
    tokens: 1000000

trainer:
  lr: 3.0e-4
  optimizer:
    type: adamw

deployment:
  type: local
  hardware: [gpu]

targets:
  - name: baseline
    description: Standard attention run
    backend: go
    system:
      topology:
        nodes:
          - id: relu
            op: activation.relu
            out: [hidden]
    runs:
      - id: train
        mode: train
        seed: 1337
        steps: 1000
        train:
          lr: 3.0e-4
`)
				experiment, err := compiler.CompileExperiment("exp.yml")
				So(err, ShouldBeNil)
				So(experiment, ShouldNotBeNil)

				So(experiment.Datasets, ShouldHaveLength, 1)
				So(experiment.Datasets[0]["name"], ShouldEqual, "train_set")

				So(experiment.Trainer["lr"], ShouldEqual, 3.0e-4)
				So(experiment.Deployment["type"], ShouldEqual, "local")

				So(experiment.Targets, ShouldHaveLength, 1)
				target := experiment.Targets[0]
				So(target.Name, ShouldEqual, "baseline")
				So(target.Graph, ShouldNotBeNil)

				So(target.Runs, ShouldHaveLength, 1)
				So(target.Runs[0].ID, ShouldEqual, "train")
				So(target.Runs[0].Mode, ShouldEqual, "train")
				So(target.Runs[0].Seed, ShouldEqual, 1337)
				So(target.Runs[0].Steps, ShouldEqual, 1000)
			})

			Convey("It should compile multiple targets each with their own graph", func() {
				write("multi.yml", `
targets:
  - name: standard
    system:
      topology:
        nodes:
          - id: relu
            op: activation.relu
            out: [h]
    runs: []
  - name: variant
    system:
      topology:
        nodes:
          - id: gelu
            op: activation.gelu
            out: [h]
    runs: []
`)
				experiment, err := compiler.CompileExperiment("multi.yml")
				So(err, ShouldBeNil)
				So(experiment.Targets, ShouldHaveLength, 2)
				So(experiment.Targets[0].Name, ShouldEqual, "standard")
				So(experiment.Targets[1].Name, ShouldEqual, "variant")
				So(experiment.Targets[0].Graph, ShouldNotBeNil)
				So(experiment.Targets[1].Graph, ShouldNotBeNil)
			})

			Convey("It should support a topology included with variables", func() {
				write("block/linear.yml", "nodes:\n  - id: proj\n    op: projection.linear\n    config:\n      in_features: ${include.dim}\n      out_features: ${include.dim}\n    out: [h]\n")
				write("exp2.yml", `
targets:
  - name: proj_run
    system:
      topology:
        include: block.linear
        variables:
          dim: 64
    runs: []
`)
				experiment, err := compiler.CompileExperiment("exp2.yml")
				So(err, ShouldBeNil)
				So(experiment.Targets[0].Graph, ShouldNotBeNil)
			})

			Convey("It should reject malformed targets instead of skipping them", func() {
				write("bad-target.yml", `
targets:
  - not-a-map
`)

				experiment, err := compiler.CompileExperiment("bad-target.yml")

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "targets[0] must be a mapping")
				So(experiment, ShouldBeNil)
			})

			Convey("It should reject missing target topology", func() {
				write("missing-topology.yml", `
targets:
  - name: broken
    system: {}
`)

				experiment, err := compiler.CompileExperiment("missing-topology.yml")

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "targets[0].system.topology is required")
				So(experiment, ShouldBeNil)
			})

			Convey("It should reject malformed runs instead of skipping them", func() {
				write("bad-run.yml", `
targets:
  - name: broken
    system:
      topology:
        nodes:
          - id: relu
            op: activation.relu
            out: [h]
    runs:
      - not-a-map
`)

				experiment, err := compiler.CompileExperiment("bad-run.yml")

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "targets[0].runs[0] must be a mapping")
				So(experiment, ShouldBeNil)
			})

			Convey("It should reject non-integer run values", func() {
				write("bad-run-int.yml", `
targets:
  - name: broken
    system:
      topology:
        nodes:
          - id: relu
            op: activation.relu
            out: [h]
    runs:
      - id: train
        steps: 1.5
`)

				experiment, err := compiler.CompileExperiment("bad-run-int.yml")

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "targets[0].runs[0].steps")
				So(err.Error(), ShouldContainSubstring, "must be an integer")
				So(experiment, ShouldBeNil)
			})

			Convey("It should reject malformed datasets", func() {
				write("bad-dataset.yml", `
datasets:
  - not-a-map
targets:
  - name: broken
    system:
      topology:
        nodes:
          - id: relu
            op: activation.relu
            out: [h]
`)

				experiment, err := compiler.CompileExperiment("bad-dataset.yml")

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "datasets[0] must be a mapping")
				So(experiment, ShouldBeNil)
			})

			Convey("It should reject missing required operation dimensions", func() {
				write("bad-op.yml", `
targets:
  - name: broken
    system:
      topology:
        nodes:
          - id: proj
            op: projection.linear
            config:
              in_features: 64
            out: [h]
`)

				experiment, err := compiler.CompileExperiment("bad-op.yml")

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "projection.linear config.out_features")
				So(err.Error(), ShouldContainSubstring, "missing required integer")
				So(experiment, ShouldBeNil)
			})
		})
	})
}

func BenchmarkCompiler_CompileExperiment(b *testing.B) {
	root := b.TempDir()
	content := `
targets:
  - name: bench
    system:
      topology:
        nodes:
          - id: relu
            op: activation.relu
            out: [h]
    runs:
      - id: train
        mode: train
        seed: 1
        steps: 100
`
	_ = os.MkdirAll(root, 0o755)
	_ = os.WriteFile(filepath.Join(root, "bench.yml"), []byte(content), 0o644)

	compiler := NewCompiler(root)

	b.ResetTimer()

	for b.Loop() {
		_, _ = compiler.CompileExperiment("bench.yml")
	}
}
