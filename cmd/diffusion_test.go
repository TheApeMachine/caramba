package cmd

import (
	"io"
	"testing"

	"github.com/smartystreets/goconvey/convey"
)

func TestDiffusionPromptReader(t *testing.T) {
	convey.Convey("Given diffusion prompt arguments", t, func() {
		reader := diffusionPromptReader([]string{"An", "elephant", "playing", "chess"})
		payload, err := io.ReadAll(reader)

		convey.So(err, convey.ShouldBeNil)
		convey.So(string(payload), convey.ShouldEqual, "An elephant playing chess\n")
	})

	convey.Convey("Given no diffusion prompt arguments", t, func() {
		reader := diffusionPromptReader(nil)

		convey.So(reader, convey.ShouldBeNil)
	})
}
