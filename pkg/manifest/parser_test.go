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

			Convey("It should return an error for a missing file", func() {
				_, err := parser.Parse("nonexistent.yml")
				So(err, ShouldNotBeNil)
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

	for repeat := 0; repeat < b.N; repeat++ {
		_, _ = parser.Parse("bench.yml")
	}
}
