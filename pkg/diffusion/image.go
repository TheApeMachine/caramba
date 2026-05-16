package diffusion

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
)

type LatentImage struct {
	Width    int
	Height   int
	Channels int
	Values   []float64
}

type RGBImage struct {
	Width  int
	Height int
	Values []float64
}

func WriteRGBImage(path string, rgbImage RGBImage) error {
	if rgbImage.Width <= 0 || rgbImage.Height <= 0 {
		return fmt.Errorf("diffusion: RGB image dimensions must be positive")
	}

	expected := 3 * rgbImage.Width * rgbImage.Height

	if len(rgbImage.Values) != expected {
		return fmt.Errorf(
			"diffusion: RGB image expected %d values, got %d",
			expected,
			len(rgbImage.Values),
		)
	}

	output := image.NewRGBA(image.Rect(0, 0, rgbImage.Width, rgbImage.Height))

	for row := 0; row < rgbImage.Height; row++ {
		for column := 0; column < rgbImage.Width; column++ {
			offset := row*rgbImage.Width + column
			output.SetRGBA(column, row, color.RGBA{
				R: rgbScale(rgbImage.Values[offset]),
				G: rgbScale(rgbImage.Values[rgbImage.Width*rgbImage.Height+offset]),
				B: rgbScale(rgbImage.Values[2*rgbImage.Width*rgbImage.Height+offset]),
				A: 255,
			})
		}
	}

	file, err := os.Create(path)

	if err != nil {
		return fmt.Errorf("diffusion: create %s: %w", path, err)
	}

	defer file.Close()

	if err := png.Encode(file, output); err != nil {
		return fmt.Errorf("diffusion: encode png: %w", err)
	}

	return nil
}

func WriteLatentPreview(path string, latentImage LatentImage) error {
	if latentImage.Width <= 0 || latentImage.Height <= 0 {
		return fmt.Errorf("diffusion: latent preview dimensions must be positive")
	}

	if latentImage.Channels < 3 {
		return fmt.Errorf("diffusion: latent preview requires at least 3 channels")
	}

	expected := latentImage.Width * latentImage.Height * latentImage.Channels

	if len(latentImage.Values) != expected {
		return fmt.Errorf(
			"diffusion: latent preview expected %d values, got %d",
			expected,
			len(latentImage.Values),
		)
	}

	output := image.NewRGBA(image.Rect(0, 0, latentImage.Width, latentImage.Height))
	scale := latentScale(latentImage.Values, latentImage.Channels)

	for row := 0; row < latentImage.Height; row++ {
		for column := 0; column < latentImage.Width; column++ {
			offset := (row*latentImage.Width + column) * latentImage.Channels
			output.SetRGBA(column, row, color.RGBA{
				R: scale(latentImage.Values[offset]),
				G: scale(latentImage.Values[offset+1]),
				B: scale(latentImage.Values[offset+2]),
				A: 255,
			})
		}
	}

	file, err := os.Create(path)

	if err != nil {
		return fmt.Errorf("diffusion: create %s: %w", path, err)
	}

	defer file.Close()

	if err := png.Encode(file, output); err != nil {
		return fmt.Errorf("diffusion: encode png: %w", err)
	}

	return nil
}

func rgbScale(value float64) uint8 {
	scaled := (value + 1) * 127.5

	if scaled < 0 {
		scaled = 0
	}

	if scaled > 255 {
		scaled = 255
	}

	return uint8(math.Round(scaled))
}

func latentScale(values []float64, channels int) func(float64) uint8 {
	minValue := math.Inf(1)
	maxValue := math.Inf(-1)

	for index, value := range values {
		if index%channels >= 3 {
			continue
		}

		minValue = math.Min(minValue, value)
		maxValue = math.Max(maxValue, value)
	}

	if !isFiniteRange(minValue, maxValue) {
		return func(float64) uint8 { return 127 }
	}

	spread := maxValue - minValue

	return func(value float64) uint8 {
		normalized := (value - minValue) / spread

		if normalized < 0 {
			normalized = 0
		}

		if normalized > 1 {
			normalized = 1
		}

		return uint8(math.Round(normalized * 255))
	}
}

func isFiniteRange(minValue float64, maxValue float64) bool {
	return !math.IsInf(minValue, 0) &&
		!math.IsInf(maxValue, 0) &&
		!math.IsNaN(minValue) &&
		!math.IsNaN(maxValue) &&
		maxValue > minValue
}
