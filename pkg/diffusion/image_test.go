package diffusion

import (
	"image/png"
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestWriteLatentPreview(test *testing.T) {
	Convey("Given latent values with at least three channels", test, func() {
		path := filepath.Join(test.TempDir(), "preview.png")

		Convey("It should write a PNG preview", func() {
			err := WriteLatentPreview(path, LatentImage{
				Width:    2,
				Height:   1,
				Channels: 3,
				Values:   []float64{-1, 0, 1, 1, 0, -1},
			})
			So(err, ShouldBeNil)

			file, err := os.Open(path)
			So(err, ShouldBeNil)
			defer file.Close()

			image, err := png.Decode(file)
			So(err, ShouldBeNil)
			So(image.Bounds().Dx(), ShouldEqual, 2)
			So(image.Bounds().Dy(), ShouldEqual, 1)
		})
	})
}

func BenchmarkWriteLatentPreview(benchmark *testing.B) {
	path := filepath.Join(benchmark.TempDir(), "preview.png")
	values := make([]float64, 64*64*3)

	for index := range values {
		values[index] = float64(index%255) / 255
	}

	for benchmark.Loop() {
		if err := WriteLatentPreview(path, LatentImage{
			Width:    64,
			Height:   64,
			Channels: 3,
			Values:   values,
		}); err != nil {
			benchmark.Fatal(err)
		}
	}
}
