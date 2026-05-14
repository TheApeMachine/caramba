package model

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Freeze marks weight groups as frozen or unfrozen in the WeightMap by
storing a sentinel key. Downstream training nodes check this before
computing or applying gradients.

The convention: a key "<pattern>.frozen" with value []float64{1} means
all weights matching <pattern> are frozen; []float64{0} means unfrozen.

Config keys:

	source   — must match the Loader node's source key
	pattern  — glob pattern of layers to freeze/unfreeze
	except   — optional glob pattern of layers to exclude from the operation
	frozen   — true (default) to freeze, false to unfreeze
*/
type Freeze struct {
	source  string
	pattern string
	except  string
	frozen  bool
}

/*
NewFreeze creates a Freeze node.
*/
func NewFreeze(source, pattern, except string, frozen bool) *Freeze {
	return &Freeze{
		source:  source,
		pattern: pattern,
		except:  except,
		frozen:  frozen,
	}
}

/*
Forward applies freeze/unfreeze markers to all matching weights.
Input: data[0] = trigger token.
Output: number of weight groups affected.
*/
func (freeze *Freeze) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.Err(); err != nil {
		return nil, err
	}

	if stateDict.Source == "" || stateDict.Pattern == "" {
		return nil, fmt.Errorf("model.freeze: Source and Pattern are required")
	}

	weights, ok := globalRegistry.Get(stateDict.Source)

	if !ok {
		return nil, fmt.Errorf("model.freeze: source %q not loaded", stateDict.Source)
	}

	selected := weights.Select(stateDict.Pattern)
	affected := 0

	sentinel := []float64{0}

	if stateDict.Frozen {
		sentinel = []float64{1}
	}

	for key := range selected {
		if stateDict.Except != "" && matchGlobPrefix(stateDict.Except, key) { //nolint — prefix match intentional
			continue
		}

		weights[key+".frozen"] = sentinel
		affected++
	}

	globalRegistry.store(stateDict.Source, weights)
	stateDict.SetOperationOutput([]float64{float64(affected)})

	return stateDict, nil
}

/*
IsFrozen returns true if the given weight key is marked frozen in weights.
Used by training nodes before applying gradient updates.
*/
func IsFrozen(weights WeightMap, key string) bool {
	sentinel, ok := weights[key+".frozen"]

	return ok && len(sentinel) > 0 && sentinel[0] == 1
}
