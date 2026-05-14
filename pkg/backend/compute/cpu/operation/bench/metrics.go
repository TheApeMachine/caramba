package bench

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Accuracy accumulates per-sample argmax accuracy in the supplied state dict.
*/
type Accuracy struct{}

func NewAccuracy() *Accuracy { return &Accuracy{} }

func (accuracy *Accuracy) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("bench.accuracy", 2); err != nil {
		return nil, err
	}

	predicted, targets := stateDict.Inputs[0], stateDict.Inputs[1]

	if len(predicted) == 0 || len(predicted) != len(targets) {
		return nil, fmt.Errorf("bench.accuracy: prediction and target lengths must match and be non-zero")
	}

	stateDict.Total++

	if argmax(predicted) == argmax(targets) {
		stateDict.Correct++
	}

	stateDict.EnsureOperationOutLen(1)
	stateDict.Out[0] = float64(stateDict.Correct) / float64(stateDict.Total)

	return stateDict, nil
}

/*
Perplexity accumulates per-sample cross-entropy in the supplied state dict.
*/
type Perplexity struct{}

func NewPerplexity() *Perplexity { return &Perplexity{} }

func (perplexity *Perplexity) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("bench.perplexity", 2); err != nil {
		return nil, err
	}

	probs, targets := stateDict.Inputs[0], stateDict.Inputs[1]

	if len(probs) == 0 || len(probs) != len(targets) {
		return nil, fmt.Errorf("bench.perplexity: probability and target lengths must match and be non-zero")
	}

	crossEntropy := 0.0

	for index, target := range targets {
		crossEntropy -= math.Log(probs[index]+1e-9) * target
	}

	stateDict.Total++
	stateDict.Sum += crossEntropy
	stateDict.EnsureOperationOutLen(1)
	stateDict.Out[0] = math.Exp(stateDict.Sum / float64(stateDict.Total))

	return stateDict, nil
}

/*
F1 accumulates binary classification TP/FP/FN in the supplied state dict.
*/
type F1 struct{}

func NewF1() *F1 { return &F1{} }

func (f1 *F1) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("bench.f1", 2); err != nil {
		return nil, err
	}

	predicted, targets := stateDict.Inputs[0], stateDict.Inputs[1]

	if len(predicted) != len(targets) {
		return nil, fmt.Errorf("bench.f1: prediction and target lengths must match")
	}

	threshold := 0.5

	for index := range predicted {
		pred := predicted[index] >= threshold
		actual := targets[index] >= threshold

		switch {
		case pred && actual:
			stateDict.TP++
		case pred && !actual:
			stateDict.FP++
		case !pred && actual:
			stateDict.FN++
		}
	}

	precision := stateDict.TP / (stateDict.TP + stateDict.FP + 1e-9)
	recall := stateDict.TP / (stateDict.TP + stateDict.FN + 1e-9)
	stateDict.EnsureOperationOutLen(1)
	stateDict.Out[0] = 2 * precision * recall / (precision + recall + 1e-9)

	return stateDict, nil
}

func argmax(xs []float64) int {
	return argmaxImpl(xs)
}
