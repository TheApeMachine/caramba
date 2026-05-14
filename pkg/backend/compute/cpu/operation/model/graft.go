package model

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Graft is a tap point on a named layer. In read mode it emits the layer's
current weights as a tensor output. In read_write mode it also accepts an
injection input that is added back into the weights — enabling activation
steering, representation editing, and probing exactly as described in the
LARQL mechanistic interpretability model.

The LARQL insight: a graft with read_write is activation patching.
Wire a trained steering vector or probe output to data[1] and it will be
added to the layer's weight slice before being stored back.

Config keys:

	source  — must match the Loader node's source key
	at      — dot-path or glob pattern of the layer to tap
	mode    — read | read_write (default: read)
*/
type Graft struct {
	source string
	at     string
	mode   string
}

/*
NewGraft creates a Graft node.
*/
func NewGraft(source, at, mode string) *Graft {
	if mode == "" {
		mode = "read"
	}

	return &Graft{source: source, at: at, mode: mode}
}

/*
Forward reads the targeted layer weights and, in read_write mode, adds
data[1] (the injection vector) back before storing. Returns the (possibly
patched) layer weights as the output tensor.

Inputs:

	data[0] — trigger token (from Loader or Surgery)
	data[1] — (read_write only) injection vector to add to the layer

Output:

	flat float64 slice of the targeted layer's weights
*/
func (graft *Graft) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.Err(); err != nil {
		return nil, err
	}

	if stateDict.Source == "" || stateDict.At == "" {
		return nil, fmt.Errorf("model.graft: Source and At are required")
	}

	weights, ok := globalRegistry.Get(stateDict.Source)

	if !ok {
		return nil, fmt.Errorf("model.graft: source %q not loaded", stateDict.Source)
	}

	selected := weights.Select(stateDict.At)

	flat := flatten(selected)

	if stateDict.Mode != "read_write" || len(stateDict.Inputs) < 2 || len(stateDict.Inputs[1]) == 0 {
		stateDict.SetOperationOutput(flat)

		return stateDict, nil
	}

	injection := stateDict.Inputs[1]
	patched := make([]float64, len(flat))
	copy(patched, flat)

	for idx := range patched {
		if idx < len(injection) {
			patched[idx] += injection[idx]
		}
	}

	// Write the patched values back into the matching keys.
	writeBack(weights, stateDict.At, patched)
	globalRegistry.store(stateDict.Source, weights)
	stateDict.SetOperationOutput(patched)

	return stateDict, nil
}

// flatten returns all values from a WeightMap as a single ordered slice.
func flatten(weights WeightMap) []float64 {
	total := 0

	for _, v := range weights {
		total += len(v)
	}

	out := make([]float64, 0, total)

	for _, v := range weights {
		out = append(out, v...)
	}

	return out
}

// writeBack distributes a flat slice back into the matching weight keys,
// preserving per-key lengths.
func writeBack(weights WeightMap, pattern string, patched []float64) {
	selected := weights.Select(pattern)
	offset := 0

	for key, val := range selected {
		end := offset + len(val)

		if end > len(patched) {
			end = len(patched)
		}

		copy(weights[key], patched[offset:end])
		offset = end
	}
}
