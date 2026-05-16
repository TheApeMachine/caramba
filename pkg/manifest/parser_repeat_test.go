package manifest

import (
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestParser_Repeat(t *testing.T) {
	Convey("Given a Parser and a manifest with a repeat block", t, func() {
		root := t.TempDir()
		parser := NewParser(root)

		content := `
nodes:
  - repeat: 3
    index: layer_idx
    template:
      - id: norm_${layer_idx}
        op: math.rmsnorm
      - id: proj_${layer_idx}
        op: projection.linear
`
		path := filepath.Join(root, "repeat.yml")
		So(os.WriteFile(path, []byte(content), 0o644), ShouldBeNil)

		Convey("When parsing the manifest", func() {
			doc, err := parser.Parse("repeat.yml")
			So(err, ShouldBeNil)

			Convey("It should expand the repeat block into a flat sequence", func() {
				nodes, ok := doc["nodes"].([]any)
				So(ok, ShouldBeTrue)
				So(len(nodes), ShouldEqual, 6)

				n0 := nodes[0].(map[string]any)
				So(n0["id"], ShouldEqual, "norm_0")
				So(n0["op"], ShouldEqual, "math.rmsnorm")

				n1 := nodes[1].(map[string]any)
				So(n1["id"], ShouldEqual, "proj_0")
				So(n1["op"], ShouldEqual, "projection.linear")

				n4 := nodes[4].(map[string]any)
				So(n4["id"], ShouldEqual, "norm_2")
				So(n4["op"], ShouldEqual, "math.rmsnorm")

				n5 := nodes[5].(map[string]any)
				So(n5["id"], ShouldEqual, "proj_2")
				So(n5["op"], ShouldEqual, "projection.linear")
			})
		})
	})
}

func TestParser_RepeatOffset(t *testing.T) {
	Convey("Given a Parser and a repeat block with a state offset", t, func() {
		root := t.TempDir()
		parser := NewParser(root)

		content := `
nodes:
  - repeat: 2
    index: block
    offset: 5
    template:
      - id: block_${block}
        in: ["state_${offset_block}"]
        out: ["state_${next_offset_block}"]
`
		path := filepath.Join(root, "repeat-offset.yml")
		So(os.WriteFile(path, []byte(content), 0o644), ShouldBeNil)

		Convey("It should keep block IDs local while offsetting state bindings", func() {
			doc, err := parser.Parse("repeat-offset.yml")
			So(err, ShouldBeNil)

			nodes, ok := doc["nodes"].([]any)
			So(ok, ShouldBeTrue)
			So(nodes, ShouldHaveLength, 2)

			first := nodes[0].(map[string]any)
			So(first["id"], ShouldEqual, "block_0")
			So(first["in"], ShouldResemble, []any{"state_5"})
			So(first["out"], ShouldResemble, []any{"state_6"})

			second := nodes[1].(map[string]any)
			So(second["id"], ShouldEqual, "block_1")
			So(second["in"], ShouldResemble, []any{"state_6"})
			So(second["out"], ShouldResemble, []any{"state_7"})
		})
	})
}
