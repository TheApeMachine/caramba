package manifest

import (
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestParser_Parse(t *testing.T) {
	Convey("Given a Parser anchored at a temp project root", t, func() {
		root := t.TempDir()

		write := func(rel, content string) {
			path := filepath.Join(root, rel)
			So(os.MkdirAll(filepath.Dir(path), 0o755), ShouldBeNil)
			So(os.WriteFile(path, []byte(content), 0o644), ShouldBeNil)
		}

		parser := NewParser(root)

		Convey("Parse", func() {
			Convey("It should decode a simple manifest", func() {
				write("master.yml", "name: test\nvalue: 42\n")
				document, err := parser.Parse("master.yml")
				So(err, ShouldBeNil)
				So(document["name"], ShouldEqual, "test")
				So(document["value"], ShouldEqual, 42)
			})

			Convey("It should interpolate ${variable} references", func() {
				write("master.yml", "variables:\n  model:\n    dim: 512\nresult: ${model.dim}\n")
				document, err := parser.Parse("master.yml")
				So(err, ShouldBeNil)
				So(document["result"], ShouldEqual, "512")
			})

			Convey("It should reject undefined variable references", func() {
				write("bad.yml", "result: ${unknown.path}\n")
				_, err := parser.Parse("bad.yml")
				So(err, ShouldNotBeNil)
			})

			Convey("It should reject a non-mapping variables block", func() {
				write("badvars.yml", "variables: not-a-map\nx: 1\n")
				_, err := parser.Parse("badvars.yml")
				So(err, ShouldNotBeNil)
			})

			Convey("It should resolve !include dot-paths", func() {
				write("block/attention.yml", "type: sdpa\nheads: 8\n")
				write("master.yml", "attn: !!include block.attention\n")
				document, err := parser.Parse("master.yml")
				So(err, ShouldBeNil)
				included, ok := document["attn"].(map[string]any)
				So(ok, ShouldBeTrue)
				So(included["type"], ShouldEqual, "sdpa")
				So(included["heads"], ShouldEqual, 8)
			})

			Convey("It should resolve parameterised includes with an include namespace", func() {
				write("block/ffn.yml", "d_model: ${include.d_model}\nbias: false\n")
				write("master.yml", "ffn:\n  include: block.ffn\n  variables:\n    d_model: 256\n")
				document, err := parser.Parse("master.yml")
				So(err, ShouldBeNil)
				included, ok := document["ffn"].(map[string]any)
				So(ok, ShouldBeTrue)
				So(included["d_model"], ShouldEqual, "256")
				So(included["bias"], ShouldEqual, false)
			})

			Convey("It should pass parent vars into parameterised includes", func() {
				write("block/proj.yml", "out: ${include.dim}\n")
				write("master.yml", "variables:\n  model:\n    dim: 128\nproj:\n  include: block.proj\n  variables:\n    dim: ${model.dim}\n")
				document, err := parser.Parse("master.yml")
				So(err, ShouldBeNil)
				included, ok := document["proj"].(map[string]any)
				So(ok, ShouldBeTrue)
				So(included["out"], ShouldEqual, "128")
			})

			Convey("It should return an error for a missing file", func() {
				_, err := parser.Parse("nonexistent.yml")
				So(err, ShouldNotBeNil)
			})

			Convey("It should reject paths outside the parser root", func() {
				_, err := parser.Parse("../outside.yml")

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "escapes parser root")
			})

			Convey("It should reject include cycles", func() {
				write("a.yml", "next: !!include b\n")
				write("b.yml", "next: !!include a\n")

				_, err := parser.Parse("a.yml")

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "include cycle detected")
			})

			Convey("It should reject repeat counts above the configured limit", func() {
				write("repeat-limit.yml", "nodes:\n  - repeat: 4097\n    template:\n      id: too_many_${i}\n")

				_, err := parser.Parse("repeat-limit.yml")

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "repeat count 4097 exceeds limit")
			})

			Convey("ParseBytes should resolve variables, repeats, and includes like Parse", func() {
				write("block/value.yml", "name: ${include.name}\n")
				content := []byte(`
variables:
  model:
    name: alpha
items:
  - repeat: 2
    index: item
    template:
      - id: node_${item}
include_result:
  include: block.value
  variables:
    name: ${model.name}
`)
				write("bytes.yml", string(content))

				fromFile, err := parser.Parse("bytes.yml")
				So(err, ShouldBeNil)

				fromBytes, err := parser.ParseBytes(content)
				So(err, ShouldBeNil)

				So(fromBytes, ShouldResemble, fromFile)
			})
		})
	})
}

func BenchmarkParser_Parse(b *testing.B) {
	root := b.TempDir()
	content := "variables:\n  model:\n    dim: 512\nresult: ${model.dim}\n"
	benchPath := filepath.Join(root, "bench.yml")

	_ = os.WriteFile(benchPath, []byte(content), 0o644)

	parser := NewParser(root)

	b.ResetTimer()

	for b.Loop() {
		_, _ = parser.Parse("bench.yml")
	}
}
