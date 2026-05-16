package activation

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
SwiGLU splits the final dimension of each row as gates|values.
Each output lane is gate * sigmoid(gate) * value.
*/
type SwiGLU struct{}

func NewSwiGLU() *SwiGLU {
	return &SwiGLU{}
}

func (swiglu *SwiGLU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("activation.swiglu"); err != nil {
		return nil, err
	}

	input := stateDict.Inputs[0]
	shape := stateDict.OperationShape()
	inputWidth := len(input)

	if len(shape) > 0 {
		inputWidth = shape[len(shape)-1]
	}

	if inputWidth <= 0 || inputWidth%2 != 0 {
		return nil, fmt.Errorf("activation.swiglu: final dimension must be positive and even, got %d", inputWidth)
	}

	if len(input)%inputWidth != 0 {
		return nil, fmt.Errorf(
			"activation.swiglu: input length %d is not divisible by final dimension %d",
			len(input),
			inputWidth,
		)
	}

	outputWidth := inputWidth / 2
	rows := len(input) / inputWidth
	stateDict.SetOperationOutput(make([]float64, rows*outputWidth))

	for rowIndex := range rows {
		inputOffset := rowIndex * inputWidth
		outputOffset := rowIndex * outputWidth
		swigluKernel(
			stateDict.Out[outputOffset:outputOffset+outputWidth],
			input[inputOffset:inputOffset+inputWidth],
		)
	}

	return stateDict, nil
}
