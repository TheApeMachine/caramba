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
