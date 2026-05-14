package model

import (
	"fmt"
	"sort"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Surgery mutates a loaded model's WeightMap by inserting, replacing, or
removing layers. It operates on the WeightRegistry so changes are visible
to all downstream nodes sharing the same source.

Config keys:

	source  — must match the Loader node's source key
	op      — insert | replace | remove
	at      — exact dot-path or index (e.g. "transformer.h.4" or "4")
	after   — dot-path or index; used by insert to position the new layer
	layer   — flat float64 slice to insert/replace (omit for remove)
	name    — name to assign the inserted/replaced layer
*/
type Surgery struct {
	source string
	op     string
	at     string
	after  string
	name   string
	layer  []float64
}

/*
NewSurgery creates a Surgery node.
*/
func NewSurgery(source, op, at, after, name string, layer []float64) *Surgery {
	return &Surgery{
		source: source,
		op:     op,
		at:     at,
		after:  after,
		name:   name,
		layer:  layer,
	}
}

/*
Forward applies the surgery operation to the WeightMap held in the registry.
Inputs: data[0] = trigger (output of Loader or prior Surgery node).
Output: passthrough trigger token so nodes can chain.
*/
func (surgery *Surgery) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.Err(); err != nil {
		return nil, err
	}

	if stateDict.Source == "" || stateDict.Op == "" {
		return nil, fmt.Errorf("model.surgery: Source and Op are required")
	}

	weights, ok := globalRegistry.Get(stateDict.Source)

	if !ok {
		return nil, fmt.Errorf("model.surgery: source %q not loaded", stateDict.Source)
	}

	operation := &Surgery{
		source: stateDict.Source,
		op:     stateDict.Op,
		at:     stateDict.At,
		after:  stateDict.After,
		name:   stateDict.Name,
		layer:  stateDict.Layer,
	}

	var err error

	switch operation.op {
	case "insert":
		err = operation.applyInsert(weights)
	case "replace":
		err = operation.applyReplace(weights)
	case "remove":
		err = operation.applyRemove(weights)
	default:
		err = fmt.Errorf("model.surgery: unknown op %q", operation.op)
	}

	if err != nil {
		return nil, err
	}

	globalRegistry.store(stateDict.Source, weights)

	if len(stateDict.Inputs) > 0 {
		stateDict.SetOperationOutput(stateDict.Inputs[0])

		return stateDict, nil
	}

	stateDict.SetOperationOutput([]float64{float64(len(weights))})

	return stateDict, nil
}

func (surgery *Surgery) applyInsert(weights WeightMap) error {
	if surgery.name == "" {
		return fmt.Errorf("model.surgery: insert requires a name")
	}

	if surgery.after != "" {
		// Shift all keys numerically after the target to make room.
		surgery.shiftLayersAfter(weights, surgery.after)
	}

	weights[surgery.name] = surgery.layer

	return nil
}

func (surgery *Surgery) applyReplace(weights WeightMap) error {
	target, err := surgery.resolveTarget(weights)

	if err != nil {
		return err
	}

	delete(weights, target)

	name := surgery.name

	if name == "" {
		name = target
	}

	weights[name] = surgery.layer

	return nil
}

func (surgery *Surgery) applyRemove(weights WeightMap) error {
	target, err := surgery.resolveTarget(weights)

	if err != nil {
		return err
	}

	// Remove all keys under this prefix.
	prefix := target + "."

	for key := range weights {
		if key == target || strings.HasPrefix(key, prefix) {
			delete(weights, key)
		}
	}

	return nil
}

// resolveTarget resolves surgery.at to an exact key in weights,
// supporting both dot-path names and numeric layer indices.
func (surgery *Surgery) resolveTarget(weights WeightMap) (string, error) {
	if _, exists := weights[surgery.at]; exists {
		return surgery.at, nil
	}

	// Try numeric index — find the Nth unique top-level layer.
	layers := layerPrefixes(weights)

	idx := 0
	fmt.Sscanf(surgery.at, "%d", &idx)

	if idx >= 0 && idx < len(layers) {
		return layers[idx], nil
	}

	return "", fmt.Errorf("model.surgery: cannot resolve target %q", surgery.at)
}

// shiftLayersAfter renumbers numeric segments in keys that come after
// the given prefix, incrementing their index to make room for a new layer.
func (surgery *Surgery) shiftLayersAfter(weights WeightMap, after string) {
	layers := layerPrefixes(weights)

	afterIdx := -1

	for idx, layer := range layers {
		if layer == after {
			afterIdx = idx
			break
		}
	}

	if afterIdx < 0 {
		return
	}

	// Walk backwards so we don't collide on rename.
	for idx := len(layers) - 1; idx > afterIdx; idx-- {
		old := layers[idx]
		parts := strings.Split(old, ".")

		// Find and increment the last numeric segment.
		for segIdx := len(parts) - 1; segIdx >= 0; segIdx-- {
			var n int

			if _, err := fmt.Sscanf(parts[segIdx], "%d", &n); err == nil {
				parts[segIdx] = fmt.Sprintf("%d", n+1)
				newKey := strings.Join(parts, ".")
				surgery.renamePrefix(weights, old, newKey)

				break
			}
		}
	}
}

func (surgery *Surgery) renamePrefix(weights WeightMap, old, new string) {
	oldPrefix := old + "."

	toRename := make(map[string]string)

	for key := range weights {
		if key == old {
			toRename[key] = new
		} else if strings.HasPrefix(key, oldPrefix) {
			toRename[key] = new + key[len(old):]
		}
	}

	for old, new := range toRename {
		weights[new] = weights[old]
		delete(weights, old)
	}
}

// layerPrefixes returns deduplicated layer paths up to and including the
// first numeric segment, sorted. E.g. "transformer.h.0.attn.q" → "transformer.h.0".
func layerPrefixes(weights WeightMap) []string {
	seen := make(map[string]struct{})

	for key := range weights {
		parts := strings.Split(key, ".")
		prefix := layerPrefix(parts)
		seen[prefix] = struct{}{}
	}

	layers := make([]string, 0, len(seen))

	for k := range seen {
		layers = append(layers, k)
	}

	sort.Strings(layers)

	return layers
}

// layerPrefix returns the path up to and including the first numeric segment,
// or the full path if none is found.
func layerPrefix(parts []string) string {
	for idx, part := range parts {
		var n int

		if _, err := fmt.Sscanf(part, "%d", &n); err == nil {
			return strings.Join(parts[:idx+1], ".")
		}
	}

	return strings.Join(parts, ".")
}
