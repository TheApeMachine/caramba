package runtime

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"

	"github.com/theapemachine/caramba/pkg/config"
	"github.com/theapemachine/caramba/pkg/tokenizer"
	manifestruntime "github.com/theapemachine/manifesto/runtime"
)

/*
CarambaHostOps implements manifest program IO and tokenizer hooks for caramba.
*/
type CarambaHostOps struct {
	hubConfig *config.HubConfig
}

/*
NewCarambaHostOps constructs host operations backed by pkg/hub and pkg/tokenizer.
*/
func NewCarambaHostOps(hubConfig *config.HubConfig) *CarambaHostOps {
	if hubConfig == nil {
		hubConfig = config.NewHubConfig()
	}

	return &CarambaHostOps{hubConfig: hubConfig}
}

func (hostOps *CarambaHostOps) ReadLine(ctx context.Context) (string, error) {
	_ = ctx

	return "", fmt.Errorf("io.read_line: use stdin via executor options")
}

func (hostOps *CarambaHostOps) EmitToken(ctx context.Context, tokenID int) error {
	_, err := fmt.Fprintf(os.Stdout, "%d ", tokenID)

	return err
}

func (hostOps *CarambaHostOps) Encode(
	ctx context.Context,
	request manifestruntime.EncodeRequest,
) ([]int, error) {
	_ = ctx

	source := tokenizer.Source{
		Source:   request.Tokenizer,
		Cache:    hostOps.hubConfig.CacheDir,
		Revision: "main",
	}

	if request.Tokenizer == "" {
		return nil, fmt.Errorf("tokenizer.encode: tokenizer name is required")
	}

	artifact, err := tokenizer.Load(ctx, source)

	if err != nil {
		return nil, err
	}

	return artifact.Tokenizer.Encode(request.Text)
}

func (hostOps *CarambaHostOps) WriteImage(
	ctx context.Context,
	request manifestruntime.WriteImageRequest,
) error {
	_ = ctx

	values, ok := request.Tensor.([]float32)

	if !ok {
		return fmt.Errorf("io.write_image: tensor must be []float32")
	}

	width := request.Width
	height := request.Height

	if width <= 0 {
		width = 64
	}

	if height <= 0 {
		height = 64
	}

	canvas := image.NewRGBA(image.Rect(0, 0, width, height))
	pixelCount := width * height

	for index := 0; index < pixelCount && index*3+2 < len(values); index++ {
		offset := index * 3
		canvas.Set(
			index%width,
			index/width,
			color.RGBA{
				R: floatToByte(values[offset]),
				G: floatToByte(values[offset+1]),
				B: floatToByte(values[offset+2]),
				A: 255,
			},
		)
	}

	path := request.Path

	if path == "" {
		path = "out.png"
	}

	file, err := os.Create(filepath.Clean(path))

	if err != nil {
		return err
	}

	defer file.Close()

	return png.Encode(file, canvas)
}

func floatToByte(value float32) uint8 {
	if value < -1 {
		return 0
	}

	if value > 1 {
		return 255
	}

	return uint8((value + 1) * 127.5)
}

var _ manifestruntime.HostOps = (*CarambaHostOps)(nil)
