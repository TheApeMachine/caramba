package tokenizer

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	runtime "github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
Encode converts configured text into token IDs.
*/
type Encode struct{}

/*
NewEncode instantiates a tokenizer encode operation.
*/
func NewEncode() *Encode {
	return &Encode{}
}

/*
Forward emits token IDs as float64 values so the compute graph can feed them
directly into embedding.token.
*/
func (encode *Encode) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.Err(); err != nil {
		return nil, err
	}

	if stateDict.Text == "" {
		return nil, fmt.Errorf("tokenizer.encode: text is required")
	}

	artifact, err := runtime.Load(context.Background(), tokenizerSource(stateDict))

	if err != nil {
		return nil, err
	}

	tokenIDs, err := artifact.Tokenizer.Encode(stateDict.Text)

	if err != nil {
		return nil, err
	}

	tokenIDs, err = applyTokenIDShape(stateDict, tokenIDs)

	if err != nil {
		return nil, err
	}

	output := make([]float64, len(tokenIDs))

	for index, tokenID := range tokenIDs {
		output[index] = float64(tokenID)
	}

	stateDict.WithShape([]int{1, len(output)})
	stateDict.SetOperationOutput(output)

	return stateDict, nil
}
