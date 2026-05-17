package io

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"

	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/state"
)

/*
WriteImage writes a tensor of floats to disk as a PNG. The tensor
layout is configured by Config keys:

  width    int  — image width
  height   int  — image height
  channels int  — values per pixel (1 for grayscale, 3 for RGB)
  path     string — destination file path
  layout   string — "channel_planar" or "channel_interleaved" (default: channel_interleaved)
  range    string — "unit" (clip to [0,1]) or "neg_one_one" (map [-1,1] to [0,1]); default unit

The op accepts either a *state.Tensor or a []float64 in
Inputs["image"]. PNG is the only supported format in this iteration.
*/
type WriteImage struct{}

func (WriteImage) Execute(execContext op.Context) error {
	step := execContext.Step()
	imageRef, ok := step.Inputs["image"]

	if !ok {
		return fmt.Errorf("io.write_image: missing input 'image'")
	}

	value, err := execContext.Resolve(imageRef)

	if err != nil {
		return err
	}

	values, err := coerceImageValues(value)

	if err != nil {
		return fmt.Errorf("io.write_image: %w", err)
	}

	width, err := requiredInt(step.Config, "width")

	if err != nil {
		return fmt.Errorf("io.write_image: %w", err)
	}

	height, err := requiredInt(step.Config, "height")

	if err != nil {
		return fmt.Errorf("io.write_image: %w", err)
	}

	channels, err := optionalIntDefault(step.Config, "channels", 3)

	if err != nil {
		return fmt.Errorf("io.write_image: %w", err)
	}

	path, err := requiredString(step.Config, "path")

	if err != nil {
		return fmt.Errorf("io.write_image: %w", err)
	}

	layout := optionalString(step.Config, "layout", "channel_interleaved")
	rangeMode := optionalString(step.Config, "range", "unit")

	expected := width * height * channels

	if len(values) != expected {
		return fmt.Errorf(
			"io.write_image: shape=[%dx%dx%d] expects %d values, got %d",
			channels, height, width, expected, len(values),
		)
	}

	output := image.NewRGBA(image.Rect(0, 0, width, height))
	scale := scalerFor(rangeMode)
	planeSize := width * height

	for row := 0; row < height; row++ {
		for column := 0; column < width; column++ {
			pixel := pixelColor(values, layout, channels, planeSize, row, column, width, scale)
			output.SetRGBA(column, row, pixel)
		}
	}

	file, err := os.Create(path)

	if err != nil {
		return fmt.Errorf("io.write_image: create %s: %w", path, err)
	}

	defer file.Close()

	if err := png.Encode(file, output); err != nil {
		return fmt.Errorf("io.write_image: encode png: %w", err)
	}

	return nil
}

func init() {
	op.Default.MustRegister("io.write_image", WriteImage{})
}

func coerceImageValues(value any) ([]float64, error) {
	switch typed := value.(type) {
	case *state.Tensor:
		return typed.Values(), nil
	case []float64:
		return typed, nil
	case []float32:
		out := make([]float64, len(typed))

		for index, element := range typed {
			out[index] = float64(element)
		}

		return out, nil
	}

	return nil, fmt.Errorf("expected tensor or []float64, got %T", value)
}

func pixelColor(
	values []float64,
	layout string,
	channels int,
	planeSize int,
	row int,
	column int,
	width int,
	scale func(float64) uint8,
) color.RGBA {
	offset := row*width + column

	switch layout {
	case "channel_planar":
		return planarPixel(values, channels, planeSize, offset, scale)
	}

	return interleavedPixel(values, channels, offset, scale)
}

func planarPixel(
	values []float64, channels int, planeSize int, offset int, scale func(float64) uint8,
) color.RGBA {
	red := scale(values[offset])
	green := red
	blue := red

	if channels >= 2 {
		green = scale(values[planeSize+offset])
	}

	if channels >= 3 {
		blue = scale(values[2*planeSize+offset])
	}

	return color.RGBA{R: red, G: green, B: blue, A: 255}
}

func interleavedPixel(
	values []float64, channels int, offset int, scale func(float64) uint8,
) color.RGBA {
	base := offset * channels
	red := scale(values[base])
	green := red
	blue := red

	if channels >= 2 {
		green = scale(values[base+1])
	}

	if channels >= 3 {
		blue = scale(values[base+2])
	}

	return color.RGBA{R: red, G: green, B: blue, A: 255}
}

func scalerFor(rangeMode string) func(float64) uint8 {
	switch rangeMode {
	case "neg_one_one":
		return func(value float64) uint8 {
			mapped := (value + 1) * 0.5

			return clampToByte(mapped)
		}
	}

	return clampToByte
}

func clampToByte(value float64) uint8 {
	if math.IsNaN(value) {
		return 0
	}

	if value < 0 {
		return 0
	}

	if value > 1 {
		return 255
	}

	return uint8(value*255 + 0.5)
}

func requiredInt(config map[string]any, key string) (int, error) {
	value, ok := config[key]

	if !ok {
		return 0, fmt.Errorf("config.%s is required", key)
	}

	return asInt(value)
}

func optionalIntDefault(config map[string]any, key string, fallback int) (int, error) {
	value, ok := config[key]

	if !ok {
		return fallback, nil
	}

	return asInt(value)
}

func requiredString(config map[string]any, key string) (string, error) {
	value, ok := config[key]

	if !ok {
		return "", fmt.Errorf("config.%s is required", key)
	}

	typed, ok := value.(string)

	if !ok {
		return "", fmt.Errorf("config.%s must be a string, got %T", key, value)
	}

	return typed, nil
}

func optionalString(config map[string]any, key string, fallback string) string {
	value, ok := config[key]

	if !ok {
		return fallback
	}

	typed, ok := value.(string)

	if !ok {
		return fallback
	}

	return typed
}
