package tokenizer

import (
	"context"
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	runtime "github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
Decode converts token IDs into text.
*/
type Decode struct{}

/*
NewDecode instantiates a tokenizer decode operation.
*/
func NewDecode() *Decode {
	return &Decode{}
}

/*
Forward decodes token IDs from Inputs[0], stores text in stateDict.Text, and
emits the decoded UTF-8 byte length as a scalar trigger.
*/
func (decode *Decode) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("tokenizer.decode", 1); err != nil {
		return nil, err
	}

	artifact, err := runtime.Load(context.Background(), tokenizerSource(stateDict))

	if err != nil {
		return nil, err
	}

	tokenIDs := make([]int, len(stateDict.Inputs[0]))

	for index, value := range stateDict.Inputs[0] {
		tokenID := int(value)

		if value != math.Trunc(value) {
			return nil, fmt.Errorf("tokenizer.decode: token[%d]=%v is not an integer", index, value)
		}

		tokenIDs[index] = tokenID
	}

	text, err := artifact.Tokenizer.Decode(tokenIDs, stateDict.SkipSpecialTokens)

	if err != nil {
		return nil, err
	}

	stateDict.Text = text
	stateDict.SetOperationOutput([]float64{float64(len([]byte(text)))})

	return stateDict, nil
}
