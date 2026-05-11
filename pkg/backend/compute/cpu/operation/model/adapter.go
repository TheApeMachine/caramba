package model

import "math"

/*
Adapter inserts a bottleneck adapter in series at a targeted layer.
A bottleneck adapter compresses the representation to a lower dimension
then projects back: out = W_up · relu(W_down · x) + x (residual).

The adapter weights are stored in the WeightMap under synthetic keys:
  <at>.adapter.down   — down-projection [dim → reduction]
  <at>.adapter.up     — up-projection   [reduction → dim]

Config keys:
  source    — must match the Loader node's source key
  at        — glob pattern of the layer to adapt
  reduction — bottleneck reduction factor (default: 16)
*/
type Adapter struct {
	source    string
	at        string
	reduction int
}

/*
NewAdapter creates an Adapter node.
*/
func NewAdapter(source, at string, reduction int) *Adapter {
	if reduction <= 0 {
		reduction = 16
	}

	return &Adapter{source: source, at: at, reduction: reduction}
}

/*
Forward inserts adapter weight matrices into the WeightMap for all matching
layers. On first call the matrices are initialised; subsequent calls are
no-ops (the weights are already present).
Input: data[0] = trigger token.
Output: number of layers adapted.
*/
func (adapter *Adapter) Forward(_ []int, data ...[]float64) []float64 {
	weights, ok := globalRegistry.Get(adapter.source)

	if !ok {
		return []float64{-1}
	}

	selected := weights.Select(adapter.at)
	inserted := 0

	for key, w := range selected {
		downKey := key + ".adapter.down"
		upKey := key + ".adapter.up"

		if _, exists := weights[downKey]; exists {
			continue
		}

		dim := len(w)
		bottleneck := max(1, dim/adapter.reduction)

		weights[downKey] = gaussianSlice(bottleneck*dim, math.Sqrt(2.0/float64(dim)))
		weights[upKey] = make([]float64, dim*bottleneck)

		inserted++
	}

	globalRegistry.store(adapter.source, weights)

	return []float64{float64(inserted)}
}

func max(a, b int) int {
	if a > b {
		return a
	}

	return b
}
