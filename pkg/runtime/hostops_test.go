package runtime

import (
	"context"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/tokenizer"
	manifestruntime "github.com/theapemachine/manifesto/runtime"
)

func TestPrepareEncodeText(test *testing.T) {
	Convey("Given tokenizer metadata with a chat template", test, func() {
		directory := test.TempDir()
		writeTestFile(test, directory, "tokenizer_config.json", `{
			"chat_template": "{{ bos_token }}<|start_header_id|>{{ role }}<|end_header_id|>",
			"bos_token": "<|begin_of_text|>"
		}`)
		writeTestFile(test, directory, "special_tokens_map.json", `{
			"eot_token": "<|eot_id|>"
		}`)

		source := tokenizer.Source{Source: directory}

		Convey("It should apply the metadata template when requested", func() {
			text, err := prepareEncodeText(context.Background(), source, " hello ", true)

			So(err, ShouldBeNil)
			So(text, ShouldEqual, "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
		})

		Convey("It should leave text unchanged when chat templates are not requested", func() {
			text, err := prepareEncodeText(context.Background(), source, " hello ", false)

			So(err, ShouldBeNil)
			So(text, ShouldEqual, " hello ")
		})
	})
}

func TestCarambaHostOpsWriteImage(test *testing.T) {
	Convey("Given a channel-planar RGB tensor", test, func() {
		directory := test.TempDir()
		path := filepath.Join(directory, "out.png")
		hostOps := &CarambaHostOps{}

		err := hostOps.WriteImage(context.Background(), manifestruntime.WriteImageRequest{
			Path:     path,
			Tensor:   []float32{1, -1, -1, 1, 0, 0},
			Width:    2,
			Height:   1,
			Channels: 3,
			Layout:   "channel_planar",
			Range:    "neg_one_one",
		})

		So(err, ShouldBeNil)

		file, err := os.Open(path)
		So(err, ShouldBeNil)
		defer file.Close()

		imageValue, err := png.Decode(file)
		So(err, ShouldBeNil)
		So(imageValue.Bounds().Dx(), ShouldEqual, 2)
		So(imageValue.Bounds().Dy(), ShouldEqual, 1)
		So(imageValue.At(0, 0), ShouldResemble, imageValue.ColorModel().Convert(color.RGBA{R: 255, G: 0, B: 127, A: 255}))
		So(imageValue.At(1, 0), ShouldResemble, imageValue.ColorModel().Convert(color.RGBA{R: 0, G: 255, B: 127, A: 255}))
	})
}

func writeTestFile(test *testing.T, directory string, name string, contents string) {
	test.Helper()

	err := os.WriteFile(filepath.Join(directory, name), []byte(contents), 0o600)

	if err != nil {
		test.Fatal(err)
	}
}
