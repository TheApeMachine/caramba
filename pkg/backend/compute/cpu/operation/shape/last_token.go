package shape

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
LastToken selects the final sequence position from a rank >= 2 tensor.
*/
type LastToken struct{}

func NewLastToken() *LastToken {
	return &LastToken{}
}

func (lastToken *LastToken) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("shape.last_token"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 2 {
		return nil, fmt.Errorf("shape.last_token: expected rank >= 2, got %d", len(shape))
	}

	sequenceLength := shape[len(shape)-2]
	featureLength := shape[len(shape)-1]

	if sequenceLength <= 0 || featureLength <= 0 {
		return nil, fmt.Errorf(
			"shape.last_token: sequence and feature dimensions must be positive, got %d and %d",
			sequenceLength,
			featureLength,
		)
	}

	outerLength, err := lastTokenOuterLength(shape)

	if err != nil {
		return nil, err
	}

	requiredInputLength := outerLength * sequenceLength * featureLength

	if len(stateDict.Inputs[0]) < requiredInputLength {
		return nil, fmt.Errorf(
			"shape.last_token: input length %d is smaller than shape length %d",
			len(stateDict.Inputs[0]),
			requiredInputLength,
		)
	}

	stateDict.EnsureOperationOutLen(outerLength * featureLength)
	lastTokenKernel(
		stateDict.Out,
		stateDict.Inputs[0],
		outerLength,
		sequenceLength,
		featureLength,
	)

	return stateDict, nil
}

func lastTokenOuterLength(shape []int) (int, error) {
	outerLength := 1

	for _, dimension := range shape[:len(shape)-2] {
		if dimension <= 0 {
			return 0, fmt.Errorf("shape.last_token: outer dimensions must be positive")
		}

		if outerLength > math.MaxInt/dimension {
			return 0, fmt.Errorf("shape.last_token: shape product overflows int")
		}

		outerLength *= dimension
	}

	return outerLength, nil
}

func lastTokenKernel(
	output []float64,
	input []float64,
	outerLength int,
	sequenceLength int,
	featureLength int,
) {
	for outerIndex := range outerLength {
		sourceStart := (outerIndex*sequenceLength + sequenceLength - 1) * featureLength
		targetStart := outerIndex * featureLength
		copy(
			output[targetStart:targetStart+featureLength],
			input[sourceStart:sourceStart+featureLength],
		)
	}
}
