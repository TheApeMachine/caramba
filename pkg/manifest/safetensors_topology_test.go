package manifest

import (
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"

	. "github.com/smartystreets/goconvey/convey"
)

func TestParser_FromSafetensors(test *testing.T) {
	Convey("Given a from_safetensors topology with embedded config", test, func() {
		fileSystem := fstest.MapFS{
			"model/architecture/registry.yml": &fstest.MapFile{
				Data: []byte(tinyArchitectureRegistry()),
			},
			"model/architecture/tiny.yml": &fstest.MapFile{
				Data: []byte(tinyArchitectureManifest()),
			},
			"master.yml": &fstest.MapFile{
				Data: []byte(`
system:
  topology:
    from_safetensors:
      architecture: TinyModel
      config:
        architectures: [TinyModel]
        hidden_size: 4
      variables:
        input_name: input
`),
			},
		}

		parser := NewParser(".").WithFS(fileSystem)

		Convey("It should expand through the architecture registry", func() {
			document, err := parser.Parse("master.yml")
			So(err, ShouldBeNil)

			topology := document["system"].(map[string]any)["topology"].(map[string]any)
			nodes := topology["nodes"].([]any)
			node := nodes[0].(map[string]any)
			config := node["config"].(map[string]any)

			So(topology["inputs"], ShouldResemble, []any{"input"})
			So(node["id"], ShouldEqual, "projection")
			So(config["in_features"], ShouldEqual, 4)
			So(config["out_features"], ShouldEqual, 8)
		})
	})

	Convey("Given a from_safetensors topology with a local sidecar config", test, func() {
		root := test.TempDir()
		writeFile := func(relativePath string, content string) {
			target := filepath.Join(root, relativePath)
			So(os.MkdirAll(filepath.Dir(target), 0o755), ShouldBeNil)
			So(os.WriteFile(target, []byte(content), 0o644), ShouldBeNil)
		}

		checkpoint := filepath.Join(root, "checkpoint")
		writeFile("model/architecture/registry.yml", tinyArchitectureRegistry())
		writeFile("model/architecture/tiny.yml", tinyArchitectureManifest())
		writeFile(
			"checkpoint/transformer/config.json",
			`{"architectures":["TinyModel"],"hidden_size":5}`,
		)
		writeFile("master.yml", `
system:
  topology:
    from_safetensors:
      source: `+checkpoint+`
      file: transformer/model.safetensors
      variables:
        input_name: tokens
`)

		parser := NewParser(root)

		Convey("It should read architectures[0] from config.json", func() {
			document, err := parser.Parse("master.yml")
			So(err, ShouldBeNil)

			topology := document["system"].(map[string]any)["topology"].(map[string]any)
			nodes := topology["nodes"].([]any)
			node := nodes[0].(map[string]any)
			config := node["config"].(map[string]any)

			So(topology["inputs"], ShouldResemble, []any{"tokens"})
			So(config["in_features"], ShouldEqual, 5)
			So(config["out_features"], ShouldEqual, 10)
		})
	})

	Convey("Given an unregistered safetensors architecture", test, func() {
		fileSystem := fstest.MapFS{
			"model/architecture/registry.yml": &fstest.MapFile{
				Data: []byte(tinyArchitectureRegistry()),
			},
			"master.yml": &fstest.MapFile{
				Data: []byte(`
system:
  topology:
    from_safetensors:
      architecture: MissingModel
      config:
        architectures: [MissingModel]
        hidden_size: 4
`),
			},
		}

		parser := NewParser(".").WithFS(fileSystem)

		Convey("It should fail before graph compilation", func() {
			_, err := parser.Parse("master.yml")
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "not registered")
		})
	})
}

func tinyArchitectureRegistry() string {
	return `
architectures:
  TinyModel:
    include: model.architecture.tiny
    variables:
      hidden_size: { config: hidden_size }
      doubled:
        product:
          - { config: hidden_size }
          - 2
`
}

func tinyArchitectureManifest() string {
	return `
inputs: ["${include.input_name}"]
nodes:
  - id: projection
    op: projection.linear
    in: ["${include.input_name}"]
    out: [projection]
    config:
      in_features: ${include.hidden_size}
      out_features: ${include.doubled}
`
}

func BenchmarkParser_FromSafetensors(benchmark *testing.B) {
	fileSystem := fstest.MapFS{
		"model/architecture/registry.yml": &fstest.MapFile{
			Data: []byte(tinyArchitectureRegistry()),
		},
		"model/architecture/tiny.yml": &fstest.MapFile{
			Data: []byte(tinyArchitectureManifest()),
		},
		"master.yml": &fstest.MapFile{
			Data: []byte(`
system:
  topology:
    from_safetensors:
      architecture: TinyModel
      config:
        architectures: [TinyModel]
        hidden_size: 4
      variables:
        input_name: input
`),
		},
	}

	parser := NewParser(".").WithFS(fileSystem)

	for benchmark.Loop() {
		if _, err := parser.Parse("master.yml"); err != nil {
			benchmark.Fatal(err)
		}
	}
}
