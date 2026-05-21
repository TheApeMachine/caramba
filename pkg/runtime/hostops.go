package runtime

import (
	"bufio"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"

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

	fmt.Print("> ")
	reader := bufio.NewReader(os.Stdin)
	text, err := reader.ReadString('\n')
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(text), nil
}

func (hostOps *CarambaHostOps) EmitToken(ctx context.Context, request manifestruntime.EmitTokenRequest) error {
	if request.Tokenizer == "" {
		_, err := fmt.Fprintf(os.Stdout, "%d ", request.TokenID)
		return err
	}

	source := tokenizer.Source{
		Source:   request.Tokenizer,
		File:     request.TokenizerFile,
		Cache:    hostOps.hubConfig.CacheDir,
		Revision: "main",
	}

	artifact, err := tokenizer.Load(ctx, source)
	if err != nil {
		return err
	}

	text, err := artifact.Tokenizer.Decode([]int{request.TokenID}, true)
	if err != nil {
		return err
	}

	_, err = fmt.Fprint(os.Stdout, text)
	return err
}

func (hostOps *CarambaHostOps) Encode(
	ctx context.Context,
	request manifestruntime.EncodeRequest,
) ([]int, error) {
	_ = ctx

	source := tokenizer.Source{
		Source:   request.Tokenizer,
		File:     request.TokenizerFile,
		Cache:    hostOps.hubConfig.CacheDir,
		Revision: "main",
	}

	if request.Tokenizer == "" {
		return nil, fmt.Errorf("tokenizer.encode: tokenizer name is required")
	}

	text, err := prepareEncodeText(ctx, source, request.Text, request.ApplyChatTemplate)

	if err != nil {
		return nil, err
	}

	artifact, err := tokenizer.Load(ctx, source)

	if err != nil {
		return nil, err
	}

	return artifact.Tokenizer.Encode(text)
}

func prepareEncodeText(
	ctx context.Context,
	source tokenizer.Source,
	text string,
	applyChatTemplate bool,
) (string, error) {
	if !applyChatTemplate {
		return text, nil
	}

	metadata, err := tokenizer.LoadMetadata(ctx, source)

	if err != nil {
		return "", err
	}

	return metadata.ApplyChatTemplate(text)
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

	channels := request.Channels

	if channels <= 0 {
		channels = 3
	}

	canvas := image.NewRGBA(image.Rect(0, 0, width, height))
	pixelCount := width * height

	for index := 0; index < pixelCount; index++ {
		red, green, blue, ok := imagePixel(values, index, pixelCount, channels, request.Layout)

		if !ok {
			break
		}

		canvas.Set(
			index%width,
			index/width,
			color.RGBA{
				R: floatToByte(red, request.Range),
				G: floatToByte(green, request.Range),
				B: floatToByte(blue, request.Range),
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

func imagePixel(
	values []float32,
	index int,
	pixelCount int,
	channels int,
	layout string,
) (float32, float32, float32, bool) {
	if layout == "channel_planar" {
		return channelPlanarPixel(values, index, pixelCount, channels)
	}

	return interleavedPixel(values, index, channels)
}

func channelPlanarPixel(
	values []float32,
	index int,
	pixelCount int,
	channels int,
) (float32, float32, float32, bool) {
	if channels < 3 || index+2*pixelCount >= len(values) {
		return 0, 0, 0, false
	}

	return values[index], values[pixelCount+index], values[2*pixelCount+index], true
}

func interleavedPixel(
	values []float32,
	index int,
	channels int,
) (float32, float32, float32, bool) {
	offset := index * channels

	if channels < 3 || offset+2 >= len(values) {
		return 0, 0, 0, false
	}

	return values[offset], values[offset+1], values[offset+2], true
}

func floatToByte(value float32, valueRange string) uint8 {
	if valueRange == "zero_one" {
		return zeroOneToByte(value)
	}

	if value < -1 {
		return 0
	}

	if value > 1 {
		return 255
	}

	return uint8((value + 1) * 127.5)
}

func zeroOneToByte(value float32) uint8 {
	if value < 0 {
		return 0
	}

	if value > 1 {
		return 255
	}

	return uint8(value * 255)
}

var _ manifestruntime.HostOps = (*CarambaHostOps)(nil)
