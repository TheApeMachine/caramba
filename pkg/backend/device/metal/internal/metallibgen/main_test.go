package main

import (
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
		args := generator.MetalArgs()

		convey.So(args, convey.ShouldResemble, []string{
			"-sdk",
			"macosx",
			"metal",
			"-c",
			filepath.Join("/workspace/metal", "add_float32.metal"),
			"-o",
			filepath.Join("/tmp/caramba-metal", "add_float32.air"),
		})
	})
}

func TestGenerator_MetallibArgs(t *testing.T) {
	convey.Convey("Given a Metal library generator", t, func() {
		generator := NewGenerator("/workspace/metal", "/tmp/caramba-metal")
		args := generator.MetallibArgs()

		convey.So(args, convey.ShouldResemble, []string{
			"-sdk",
			"macosx",
			"metallib",
			filepath.Join("/tmp/caramba-metal", "add_float32.air"),
			"-o",
			filepath.Join("/workspace/metal", "kernels.metallib"),
		})
	})
}
