package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestNewGenerator(t *testing.T) {
	convey.Convey("Given a Metal library generator", t, func() {
		generator := NewGenerator("/workspace/metal", "/tmp/caramba-metal")

		convey.So(generator.packageDir, convey.ShouldEqual, "/workspace/metal")
		convey.So(generator.tempDir, convey.ShouldEqual, "/tmp/caramba-metal")
	})
}

func TestGenerator_MetalArgs(t *testing.T) {
	convey.Convey("Given a Metal library generator", t, func() {
		generator := NewGenerator("/workspace/metal", "/tmp/caramba-metal")
		source := filepath.Join("/workspace/metal", "elementwise_float32.metal")
		args := generator.MetalArgs(source)

		convey.So(args, convey.ShouldResemble, []string{
			"-sdk",
			"macosx",
			"metal",
			"-c",
			source,
			"-o",
			filepath.Join("/tmp/caramba-metal", "elementwise_float32.air"),
		})
	})
}

func TestGenerator_MetallibArgs(t *testing.T) {
	convey.Convey("Given a Metal library generator", t, func() {
		generator := NewGenerator("/workspace/metal", "/tmp/caramba-metal")
		sources := []string{
			filepath.Join("/workspace/metal", "elementwise_float32.metal"),
			filepath.Join("/workspace/metal", "matmul_float32.metal"),
		}
		args := generator.MetallibArgs(sources)

		convey.So(args, convey.ShouldResemble, []string{
			"-sdk",
			"macosx",
			"metallib",
			filepath.Join("/tmp/caramba-metal", "elementwise_float32.air"),
			filepath.Join("/tmp/caramba-metal", "matmul_float32.air"),
			"-o",
			filepath.Join("/workspace/metal", "kernels.metallib"),
		})
	})
}

func TestGenerator_SourceFiles(t *testing.T) {
	convey.Convey("Given a package directory with Metal sources", t, func() {
		packageDir := t.TempDir()
		generator := NewGenerator(packageDir, "/tmp/caramba-metal")

		err := os.WriteFile(filepath.Join(packageDir, "zeta.metal"), []byte(""), 0600)
		convey.So(err, convey.ShouldBeNil)

		err = os.WriteFile(filepath.Join(packageDir, "alpha.metal"), []byte(""), 0600)
		convey.So(err, convey.ShouldBeNil)

		sources, err := generator.SourceFiles()
		convey.So(err, convey.ShouldBeNil)
		convey.So(sources, convey.ShouldResemble, []string{
			filepath.Join(packageDir, "alpha.metal"),
			filepath.Join(packageDir, "zeta.metal"),
		})
	})
}

func TestGenerator_AirPath(t *testing.T) {
	convey.Convey("Given a Metal source path", t, func() {
		generator := NewGenerator("/workspace/metal", "/tmp/caramba-metal")
		source := filepath.Join("/workspace/metal", "elementwise_float32.metal")

		convey.So(
			generator.AirPath(source),
			convey.ShouldEqual,
			filepath.Join("/tmp/caramba-metal", "elementwise_float32.air"),
		)
	})
}
