package tokenizer

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
	runtime "github.com/theapemachine/caramba/pkg/tokenizer"
)

func tokenizerSource(stateDict *state.Dict) runtime.Source {
	return runtime.Source{
		Source:   stateDict.Source,
		File:     stateDict.File,
		Cache:    stateDict.Cache,
		Revision: stateDict.Revision,
		RepoType: stateDict.RepoType,
	}
}

func applyTokenIDShape(
	stateDict *state.Dict,
	tokenIDs []int,
) ([]int, error) {
	if stateDict.MaxLength > 0 && len(tokenIDs) > stateDict.MaxLength {
		if !stateDict.Truncate {
			return nil, fmt.Errorf(
				"tokenizer.encode: token length %d exceeds max_length %d",
				len(tokenIDs),
				stateDict.MaxLength,
			)
		}

		tokenIDs = append([]int(nil), tokenIDs[:stateDict.MaxLength]...)
	}

	if stateDict.PadTo > 0 {
		if len(tokenIDs) > stateDict.PadTo {
			return nil, fmt.Errorf(
				"tokenizer.encode: token length %d exceeds pad_to %d",
				len(tokenIDs),
				stateDict.PadTo,
			)
		}

		padded := make([]int, stateDict.PadTo)
		copy(padded, tokenIDs)

		for index := len(tokenIDs); index < len(padded); index++ {
			padded[index] = stateDict.PadID
		}

		tokenIDs = padded
	}

	return tokenIDs, nil
}
