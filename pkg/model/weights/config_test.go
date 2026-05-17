package weights

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestResolveConfig(test *testing.T) {
	Convey("Given a component-scoped SafeTensors sidecar config", test, func() {
		root := test.TempDir()
		component := filepath.Join(root, "transformer")
		So(os.MkdirAll(component, 0o755), ShouldBeNil)
		So(
			os.WriteFile(
				filepath.Join(component, "config.json"),
				[]byte(`{"architectures":["TinyModel"],"hidden_size":7}`),
				0o644,
			),
			ShouldBeNil,
		)

		Convey("It should resolve config.json next to the requested file", func() {
			config, err := ResolveConfig(context.Background(), Source{
				Source: root,
				File:   "transformer/model.safetensors",
			})
			So(err, ShouldBeNil)
			So(config["hidden_size"], ShouldEqual, 7.0)
			So(config["architectures"], ShouldResemble, []any{"TinyModel"})
		})
	})
}

func BenchmarkResolveConfig(benchmark *testing.B) {
	root := benchmark.TempDir()
	component := filepath.Join(root, "transformer")

	if err := os.MkdirAll(component, 0o755); err != nil {
		benchmark.Fatal(err)
	}

	if err := os.WriteFile(
		filepath.Join(component, "config.json"),
		[]byte(`{"architectures":["TinyModel"],"hidden_size":7}`),
		0o644,
	); err != nil {
		benchmark.Fatal(err)
	}

	source := Source{
		Source: root,
		File:   "transformer/model.safetensors",
	}

	for benchmark.Loop() {
		if _, err := ResolveConfig(context.Background(), source); err != nil {
			benchmark.Fatal(err)
		}
	}
}
