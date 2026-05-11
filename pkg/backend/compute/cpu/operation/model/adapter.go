package model

import (
	"fmt"
	"math"
	"sort"
)

/*
Adapter inserts a bottleneck adapter in series at a targeted layer.
Forward pass: out = W_up · relu(W_down · x) + x

	W_down: [bottleneck × dim]  — compresses input
	W_up:   [dim × bottleneck]  — projects back to original dimension
	residual connection preserves representation at init (W_up is zero-init)

Weights are stored in the WeightMap as:

	<layer>.adapter.down  [bottleneck × dim]
	<layer>.adapter.up    [dim × bottleneck]

The matmul is injected so Metal/CUDA/XLA supply their kernels.
*/
type Adapter struct {
	source    string
	at        string
	reduction int
	matmul    MatMulFn
}

/*
NewAdapter creates an Adapter node using CPUMatMul.
*/
func NewAdapter(source, at string, reduction int) *Adapter {
	return NewAdapterWithMatMul(source, at, reduction, CPUMatMul)
}

/*
NewAdapterWithMatMul creates an Adapter node with an injected matmul kernel.
*/
func NewAdapterWithMatMul(source, at string, reduction int, matmul MatMulFn) *Adapter {
	if reduction <= 0 {
		reduction = 16
	}

	return &Adapter{source: source, at: at, reduction: reduction, matmul: matmul}
}

/*
Forward initialises adapter weight matrices on first call, then computes
the full bottleneck forward pass for each matched layer:

	h   = relu(W_down · x)   [bottleneck]
	out = W_up · h + x       [dim]

Input: data[0] = the activation tensor for the matched layer (flat [dim]).
Output: the adapted activation (flat [dim]).

On the first call with no input data (trigger-only mode from a Loader node),
Forward initialises the matrices and returns a token count.
*/
func (adapter *Adapter) Forward(_ []int, data ...[]float64) []float64 {
	weights, ok := globalRegistry.Get(adapter.source)

	if !ok {
		return []float64{-1}
	}

	selected := weights.Select(adapter.at)

	// Trigger-only call (no activation input): initialise matrices.
	if len(data) == 0 || len(data[0]) <= 1 {
		return adapter.initialise(weights, selected)
	}

	x := data[0]
	var out []float64
	changed := false

	keys := make([]string, 0, len(selected))

	for key := range selected {
		keys = append(keys, key)
	}

	sort.Strings(keys)

	for _, key := range keys {
		downKey := key + ".adapter.down"
		upKey := key + ".adapter.up"

		wDown, downOK := weights[downKey]
		wUp, upOK := weights[upKey]

		if !downOK || !upOK {
			adapter.insertMatrices(weights, key, len(x))
			wDown = weights[downKey]
			wUp = weights[upKey]
			changed = true
		}

		dim := len(x)
		bottleneck := max(1, dim/adapter.reduction)

		// h = W_down · x  →  [bottleneck × dim] · [dim × 1] = [bottleneck × 1]
		h, err := adapter.matmul(wDown, x, bottleneck, dim, 1)

		if err != nil {
			panic(fmt.Errorf("adapter: W_down matmul: %w", err))
		}

		reluInPlace(h)

		// out = W_up · h + x  →  [dim × bottleneck] · [bottleneck × 1] = [dim × 1]
		projected, err := adapter.matmul(wUp, h, dim, bottleneck, 1)

		if err != nil {
			panic(fmt.Errorf("adapter: W_up matmul: %w", err))
		}

		out = make([]float64, dim)

		for idx := range projected {
			out[idx] = projected[idx] + x[idx]
		}

		// chain: next layer receives this layer's output as residual
		x = out
	}

	if changed {
		globalRegistry.store(adapter.source, weights)
	}

	if out == nil {
		return data[0]
	}

	return out
}

func (adapter *Adapter) initialise(weights WeightMap, selected WeightMap) []float64 {
	inserted := 0

	for key, w := range selected {
		if _, exists := weights[key+".adapter.down"]; exists {
			continue
		}

		adapter.insertMatrices(weights, key, len(w))
		inserted++
	}

	globalRegistry.store(adapter.source, weights)

	return []float64{float64(inserted)}
}

func (adapter *Adapter) insertMatrices(weights WeightMap, key string, dim int) {
	bottleneck := max(1, dim/adapter.reduction)

	// W_down [bottleneck × dim]: gaussian init.
	weights[key+".adapter.down"] = gaussianSlice(
		bottleneck*dim, math.Sqrt(2.0/float64(dim)),
	)
	// W_up [dim × bottleneck]: zero init — residual is identity at start.
	weights[key+".adapter.up"] = make([]float64, dim*bottleneck)
}

func max(a, b int) int {
	if a > b {
		return a
	}

	return b
}
