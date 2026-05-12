package train

import (
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
	n := len(predictions)

	if n == 0 {
		return []float64{0}
	}

	diff := make([]float64, n)
	copy(diff, predictions)
	cpumath.AddScaledVec(diff, targets, -1)
	sumSq := cpumath.L2NormSq(diff)

	return []float64{sumSq / float64(n)}
}

/*
CrossEntropyLoss computes softmax cross-entropy between logits (data[0]) and
one-hot targets (data[1]). Returns a single-element slice containing the loss.
*/
type CrossEntropyLoss struct{}

func NewCrossEntropyLoss() *CrossEntropyLoss { return &CrossEntropyLoss{} }

func (cel *CrossEntropyLoss) Forward(_ []int, data ...[]float64) []float64 {
	logits, targets := data[0], data[1]

	if len(logits) == 0 {
		return []float64{0}
	}

	probs := softmax(logits)
	cpumath.AddScalarVec(probs, 1e-9)
	logProbs := make([]float64, len(probs))
	cpumath.LogVec(logProbs, probs)
	cpumath.MulVec(logProbs, logProbs, targets)
	loss := -cpumath.ReduceSum(logProbs)

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
	n := len(predictions)

	if n == 0 {
		return []float64{}
	}

	scale := 2.0 / float64(n)
	grads := make([]float64, n)
	copy(grads, predictions)
	cpumath.AddScaledVec(grads, targets, -1)
	cpumath.ScaleVec(grads, scale)

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
	cpumath.AddScaledVec(probs, targets, -1)

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
