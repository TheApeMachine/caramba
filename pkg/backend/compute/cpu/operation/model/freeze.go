package model

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
func (freeze *Freeze) Forward(_ []int, data ...[]float64) []float64 {
	weights, ok := globalRegistry.Get(freeze.source)

	if !ok {
		return []float64{-1}
	}

	selected := weights.Select(freeze.pattern)
	affected := 0

	sentinel := []float64{0}

	if freeze.frozen {
		sentinel = []float64{1}
	}

	for key := range selected {
		if freeze.except != "" && matchGlobPrefix(freeze.except, key) { //nolint — prefix match intentional
			continue
		}

		weights[key+".frozen"] = sentinel
		affected++
	}

	globalRegistry.store(freeze.source, weights)

	return []float64{float64(affected)}
}

/*
IsFrozen returns true if the given weight key is marked frozen in weights.
Used by training nodes before applying gradient updates.
*/
func IsFrozen(weights WeightMap, key string) bool {
	sentinel, ok := weights[key+".frozen"]

	return ok && len(sentinel) > 0 && sentinel[0] == 1
}
