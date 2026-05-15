package tokenizer

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	runtime "github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
Load loads a tokenizer artifact into the process-local tokenizer registry.
*/
type Load struct{}

/*
NewLoad instantiates a tokenizer loader operation.
*/
func NewLoad() *Load {
	return &Load{}
}

/*
Forward loads the tokenizer and emits its vocabulary size.
*/
func (load *Load) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.Err(); err != nil {
		return nil, err
	}

	source := tokenizerSource(stateDict)

	if source.Source == "" {
		return nil, fmt.Errorf("tokenizer.load: source is required")
	}

	artifact, err := runtime.Load(context.Background(), source)

	if err != nil {
		return nil, err
	}

	stateDict.SetOperationOutput([]float64{
		float64(artifact.Tokenizer.VocabSize()),
	})

	return stateDict, nil
}
