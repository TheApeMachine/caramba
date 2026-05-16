package rotary

import (
	"fmt"
	"math"
	"strings"
)

const (
	TypeDefault = "default"
	TypeLlama3  = "llama3"
)

/*
Config describes the inverse-frequency schedule for rotary embeddings.
*/
type Config struct {
	Base                          float64
	Type                          string
	Factor                        float64
	LowFreqFactor                 float64
	HighFreqFactor                float64
	OriginalMaxPositionEmbeddings int
}

/*
InverseFrequencies returns the RoPE inverse frequencies for one head.
*/
func (config Config) InverseFrequencies(headDim int) ([]float64, error) {
	if headDim <= 0 || headDim%2 != 0 {
		return nil, fmt.Errorf("rotary: head_dim must be positive and even, got %d", headDim)
	}

	config = config.normalized()
	frequencies := make([]float64, headDim/2)

	for pairIndex := range frequencies {
		frequencies[pairIndex] = defaultInverseFrequency(
			config.Base,
			pairIndex,
			headDim,
		)
	}

	switch strings.ToLower(config.Type) {
	case "", TypeDefault:
		return frequencies, nil
	case TypeLlama3:
		config.scaleLlama3(frequencies)

		return frequencies, nil
	default:
		return nil, fmt.Errorf("rotary: unsupported rope_type %q", config.Type)
	}
}

func (config Config) normalized() Config {
	if config.Base == 0 {
		config.Base = 10000.0
	}

	if config.Factor == 0 {
		config.Factor = 1
	}

	if config.LowFreqFactor == 0 {
		config.LowFreqFactor = 1
	}

	if config.HighFreqFactor == 0 {
		config.HighFreqFactor = 4
	}

	if config.OriginalMaxPositionEmbeddings == 0 {
		config.OriginalMaxPositionEmbeddings = 8192
	}

	return config
}

func defaultInverseFrequency(base float64, pairIndex, headDim int) float64 {
	return 1.0 / math.Pow(base, float64(2*pairIndex)/float64(headDim))
}

func (config Config) scaleLlama3(frequencies []float64) {
	lowFreqWavelength := float64(config.OriginalMaxPositionEmbeddings) /
		config.LowFreqFactor
	highFreqWavelength := float64(config.OriginalMaxPositionEmbeddings) /
		config.HighFreqFactor

	for index, frequency := range frequencies {
		wavelength := 2 * math.Pi / frequency

		if wavelength < highFreqWavelength {
			continue
		}

		if wavelength > lowFreqWavelength {
			frequencies[index] = frequency / config.Factor

			continue
		}

		smooth := (float64(config.OriginalMaxPositionEmbeddings)/wavelength -
			config.LowFreqFactor) /
			(config.HighFreqFactor - config.LowFreqFactor)

		frequencies[index] = (1-smooth)*frequency/config.Factor + smooth*frequency
	}
}
