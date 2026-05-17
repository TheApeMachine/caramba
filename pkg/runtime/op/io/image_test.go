package io

import (
	"image/png"
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"

	"github.com/theapemachine/caramba/pkg/runtime/op/optest"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

func TestWriteImageInterleaved(t *testing.T) {
	Convey("Given a 2x2 interleaved RGB image of pure red", t, func() {
		stub := optest.NewStubContext()
		stub.Scope["image"] = []float64{
			1, 0, 0, 1, 0, 0,
			1, 0, 0, 1, 0, 0,
		}

		path := filepath.Join(t.TempDir(), "out.png")
		stub.StepRef = program.Step{
			ID: "write",
			Op: "io.write_image",
			Inputs: map[string]program.ValueRef{
				"image": {Namespace: program.NamespaceLocal, Name: "image"},
			},
			Config: map[string]any{
				"width":    2,
				"height":   2,
				"channels": 3,
				"path":     path,
			},
		}

		So(WriteImage{}.Execute(stub), ShouldBeNil)

		Convey("The PNG should decode back to red pixels", func() {
			file, err := os.Open(path)
			So(err, ShouldBeNil)
			defer file.Close()

			img, err := png.Decode(file)
			So(err, ShouldBeNil)

			bounds := img.Bounds()
			So(bounds.Dx(), ShouldEqual, 2)
			So(bounds.Dy(), ShouldEqual, 2)

			red, green, blue, _ := img.At(0, 0).RGBA()
			So(red>>8, ShouldEqual, 255)
			So(green>>8, ShouldEqual, 0)
			So(blue>>8, ShouldEqual, 0)
		})
	})
}

func TestWriteImagePlanar(t *testing.T) {
	Convey("Given a 2x2 planar layout (R-plane all 1, G/B 0)", t, func() {
		stub := optest.NewStubContext()
		stub.Scope["image"] = []float64{
			1, 1, 1, 1,
			0, 0, 0, 0,
			0, 0, 0, 0,
		}

		path := filepath.Join(t.TempDir(), "out.png")
		stub.StepRef = program.Step{
			ID: "write",
			Op: "io.write_image",
			Inputs: map[string]program.ValueRef{
				"image": {Namespace: program.NamespaceLocal, Name: "image"},
			},
			Config: map[string]any{
				"width":    2,
				"height":   2,
				"channels": 3,
				"layout":   "channel_planar",
				"path":     path,
			},
		}

		So(WriteImage{}.Execute(stub), ShouldBeNil)

		file, err := os.Open(path)
		So(err, ShouldBeNil)
		defer file.Close()

		img, err := png.Decode(file)
		So(err, ShouldBeNil)

		red, green, blue, _ := img.At(1, 1).RGBA()
		So(red>>8, ShouldEqual, 255)
		So(green>>8, ShouldEqual, 0)
		So(blue>>8, ShouldEqual, 0)
	})
}

func TestWriteImageNegOneOneRange(t *testing.T) {
	Convey("Given values in [-1, 1] mapped through the range option", t, func() {
		stub := optest.NewStubContext()
		stub.Scope["image"] = []float64{
			-1, -1, -1,
			1, 1, 1,
			0, 0, 0,
		}

		path := filepath.Join(t.TempDir(), "out.png")
		stub.StepRef = program.Step{
			ID: "write",
			Op: "io.write_image",
			Inputs: map[string]program.ValueRef{
				"image": {Namespace: program.NamespaceLocal, Name: "image"},
			},
			Config: map[string]any{
				"width":    1,
				"height":   3,
				"channels": 3,
				"range":    "neg_one_one",
				"path":     path,
			},
		}

		So(WriteImage{}.Execute(stub), ShouldBeNil)

		file, err := os.Open(path)
		So(err, ShouldBeNil)
		defer file.Close()

		img, err := png.Decode(file)
		So(err, ShouldBeNil)

		topRed, _, _, _ := img.At(0, 0).RGBA()
		midRed, _, _, _ := img.At(0, 1).RGBA()
		bottomRed, _, _, _ := img.At(0, 2).RGBA()

		So(topRed>>8, ShouldEqual, 0)
		So(midRed>>8, ShouldEqual, 255)
		So(bottomRed>>8, ShouldBeBetweenOrEqual, 127, 128)
	})
}
