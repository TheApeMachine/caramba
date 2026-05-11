package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

/*
WeightMap is the in-memory representation of a loaded model's parameters.
Keys are dot-separated layer paths (e.g. "transformer.h.4.attn.c_attn")
matching the naming conventions used by safetensors and GGUF parsers.
*/
type WeightMap map[string][]float64

/*
Load reads a weight map from a JSON file on disk.
This is the native caramba checkpoint format; HuggingFace models are
converted to this format on first load via the modelscope backend.
*/
func Load(path string) (WeightMap, error) {
	data, err := os.ReadFile(path)

	if err != nil {
		return nil, fmt.Errorf("model: read %s: %w", path, err)
	}

	var weights WeightMap

	err = json.Unmarshal(data, &weights)

	if err != nil {
		return nil, fmt.Errorf("model: unmarshal %s: %w", path, err)
	}

	return weights, nil
}

/*
Save persists a WeightMap to disk as JSON.
*/
func Save(path string, weights WeightMap) error {
	err := os.MkdirAll(filepath.Dir(path), 0o755)

	if err != nil {
		return fmt.Errorf("model: mkdir: %w", err)
	}

	data, err := json.Marshal(weights)

	if err != nil {
		return fmt.Errorf("model: marshal: %w", err)
	}

	return os.WriteFile(path, data, 0o644)
}

/*
Select returns all weight keys whose dot-path matches the given glob pattern.
Supports * as a wildcard segment and ** as a multi-segment wildcard.
*/
func (weights WeightMap) Select(pattern string) WeightMap {
	out := make(WeightMap)

	for key, val := range weights {
		if matchGlobPrefix(pattern, key) {
			out[key] = val
		}
	}

	return out
}

/*
LayerNames returns the unique top-level layer names in declaration order.
E.g. "transformer.h.0.attn.c_attn" and "transformer.h.0.attn.c_proj"
both contribute "transformer.h.0" at depth 3.
*/
func (weights WeightMap) LayerNames(depth int) []string {
	seen := make(map[string]struct{})
	var ordered []string

	for key := range weights {
		parts := strings.Split(key, ".")

		if len(parts) < depth {
			continue
		}

		prefix := strings.Join(parts[:depth], ".")

		if _, exists := seen[prefix]; !exists {
			seen[prefix] = struct{}{}
			ordered = append(ordered, prefix)
		}
	}

	return ordered
}

// matchGlob returns true when pattern fully matches name (all segments consumed).
func matchGlob(pattern, name string) bool {
	return matchParts(strings.Split(pattern, "."), strings.Split(name, "."))
}

// matchGlobPrefix returns true when pattern matches name or any prefix of name —
// i.e. the key lives at or under the pattern path. Used by Freeze and Select.
func matchGlobPrefix(pattern, name string) bool {
	patParts := strings.Split(pattern, ".")
	nameParts := strings.Split(name, ".")

	return matchPartsPrefix(patParts, nameParts)
}

func matchPartsPrefix(pattern, name []string) bool {
	for len(pattern) > 0 {
		seg := pattern[0]
		pattern = pattern[1:]

		switch seg {
		case "**":
			for idx := 0; idx <= len(name); idx++ {
				if matchPartsPrefix(pattern, name[idx:]) {
					return true
				}
			}

			return false

		case "*":
			if len(name) == 0 {
				// * with no remaining name segments — treat as prefix match
				return true
			}

			name = name[1:]

		default:
			if len(name) == 0 || seg != name[0] {
				return false
			}

			name = name[1:]
		}
	}

	// Pattern exhausted — match whether or not name has remaining segments
	// (prefix semantics: the key lives under the pattern path).
	return true
}

func matchParts(pattern, name []string) bool {
	for len(pattern) > 0 {
		seg := pattern[0]
		pattern = pattern[1:]

		switch seg {
		case "**":
			// ** consumes zero or more segments.
			for idx := 0; idx <= len(name); idx++ {
				if matchParts(pattern, name[idx:]) {
					return true
				}
			}

			return false

		case "*":
			// * matches exactly one segment but the pattern may still have
			// more segments after — don't require name to be exhausted here.
			if len(name) == 0 {
				return false
			}

			name = name[1:]

		default:
			if len(name) == 0 || seg != name[0] {
				return false
			}

			name = name[1:]
		}
	}

	return len(name) == 0
}
