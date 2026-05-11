package train

import (
	gomath "math"

	cpumath "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

/*
MSELoss computes mean squared error between predictions (data[0]) and targets (data[1]).
Returns a single-element slice containing the scalar loss.
*/
type MSELoss struct{}

func NewMSELoss() *MSELoss { return &MSELoss{} }

func (mse *MSELoss) Forward(_ []int, data ...[]float64) []float64 {
	predictions, targets := data[0], data[1]
	var sum float64

	for idx := range predictions {
		diff := predictions[idx] - targets[idx]
		sum += diff * diff
	}

	return []float64{sum / float64(len(predictions))}
}

/*
CrossEntropyLoss computes softmax cross-entropy between logits (data[0]) and
one-hot targets (data[1]). Returns a single-element slice containing the loss.
*/
type CrossEntropyLoss struct{}

func NewCrossEntropyLoss() *CrossEntropyLoss { return &CrossEntropyLoss{} }

func (cel *CrossEntropyLoss) Forward(_ []int, data ...[]float64) []float64 {
	logits, targets := data[0], data[1]
	probs := softmax(logits)
	var loss float64

	for idx, prob := range probs {
		loss -= targets[idx] * gomath.Log(prob+1e-9)
	}

	return []float64{loss}
}

/*
MSEGrad computes the gradient of MSE loss with respect to predictions.
Inputs: data[0] = predictions, data[1] = targets.
*/
type MSEGrad struct{}

func NewMSEGrad() *MSEGrad { return &MSEGrad{} }

func (g *MSEGrad) Forward(_ []int, data ...[]float64) []float64 {
	predictions, targets := data[0], data[1]
	scale := 2.0 / float64(len(predictions))
	grads := make([]float64, len(predictions))

	for idx := range predictions {
		grads[idx] = scale * (predictions[idx] - targets[idx])
	}

	return grads
}

/*
CrossEntropyGrad computes d(cross_entropy)/d(logits) = softmax(logits) - targets.
Inputs: data[0] = logits, data[1] = one-hot targets.
*/
type CrossEntropyGrad struct{}

func NewCrossEntropyGrad() *CrossEntropyGrad { return &CrossEntropyGrad{} }

func (g *CrossEntropyGrad) Forward(_ []int, data ...[]float64) []float64 {
	logits, targets := data[0], data[1]

	if len(logits) == 0 {
		return []float64{}
	}

	probs := softmax(logits)

	for idx := range probs {
		probs[idx] -= targets[idx]
	}

	return probs
}

func softmax(xs []float64) []float64 {
	if len(xs) == 0 {
		return []float64{}
	}

	out := make([]float64, len(xs))
	copy(out, xs)
	cpumath.SoftmaxSlice(out)

	return out
}
