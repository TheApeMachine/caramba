package model

import (
	"fmt"
	"math"
	"sort"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
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
func (adapter *Adapter) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.Err(); err != nil {
		return nil, err
	}

	if stateDict.Source == "" || stateDict.At == "" {
		return nil, fmt.Errorf("model.adapter: Source and At are required")
	}

	weights, ok := globalRegistry.Get(stateDict.Source)

	if !ok {
		return nil, fmt.Errorf("model.adapter: source %q not loaded", stateDict.Source)
	}

	selected := weights.Select(stateDict.At)
	reduction := stateDict.Reduction

	if reduction <= 0 {
		reduction = 16
	}

	matmul := adapter.matmul

	if matmul == nil {
		matmul = CPUMatMul
	}

	// Trigger-only call (no activation input): initialise matrices.
	if len(stateDict.Inputs) == 0 || len(stateDict.Inputs[0]) <= 1 {
		inserted := initialiseAdapter(weights, selected, reduction)
		globalRegistry.store(stateDict.Source, weights)
		stateDict.SetOperationOutput([]float64{float64(inserted)})

		return stateDict, nil
	}

	x := stateDict.Inputs[0]
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
			insertAdapterMatrices(weights, key, len(x), reduction)
			wDown = weights[downKey]
			wUp = weights[upKey]
			changed = true
		}

		dim := len(x)
		bottleneck := max(1, dim/reduction)

		// h = W_down · x  →  [bottleneck × dim] · [dim × 1] = [bottleneck × 1]
		h, err := matmul(wDown, x, bottleneck, dim, 1)

		if err != nil {
			return nil, fmt.Errorf("model.adapter: W_down matmul: %w", err)
		}

		reluInPlace(h)

		// out = W_up · h + x  →  [dim × bottleneck] · [bottleneck × 1] = [dim × 1]
		projected, err := matmul(wUp, h, dim, bottleneck, 1)

		if err != nil {
			return nil, fmt.Errorf("model.adapter: W_up matmul: %w", err)
		}

		out = make([]float64, dim)

		for idx := range projected {
			out[idx] = projected[idx] + x[idx]
		}

		// chain: next layer receives this layer's output as residual
		x = out
	}

	if changed {
		globalRegistry.store(stateDict.Source, weights)
	}

	if out == nil {
		stateDict.SetOperationOutput(stateDict.Inputs[0])

		return stateDict, nil
	}

	stateDict.SetOperationOutput(out)

	return stateDict, nil
}

func initialiseAdapter(weights WeightMap, selected WeightMap, reduction int) int {
	inserted := 0

	for key, w := range selected {
		if _, exists := weights[key+".adapter.down"]; exists {
			continue
		}

		insertAdapterMatrices(weights, key, len(w), reduction)
		inserted++
	}

	return inserted
}

func insertAdapterMatrices(weights WeightMap, key string, dim, reduction int) {
	bottleneck := max(1, dim/reduction)

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
