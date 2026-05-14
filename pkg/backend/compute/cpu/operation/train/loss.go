package train

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
MSELoss computes mean squared error between predictions (data[0]) and targets (data[1]).
Returns a single-element slice containing the scalar loss.
*/
type MSELoss struct{}

func NewMSELoss() *MSELoss { return &MSELoss{} }

func (mse *MSELoss) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("train.mse_loss", 2); err != nil {
		return nil, err
	}

	predictions, targets := stateDict.Inputs[0], stateDict.Inputs[1]
	n := len(predictions)

	if len(targets) != n {
		return nil, fmt.Errorf("train.mse_loss: prediction and target lengths differ")
	}

	stateDict.EnsureOperationOutLen(1)

	if n == 0 {
		stateDict.Out[0] = 0

		return stateDict, nil
	}

	diff := make([]float64, n)
	copy(diff, predictions)
	addScaledVec(diff, targets, -1)
	sumSq := l2NormSq(diff)
	stateDict.Out[0] = sumSq / float64(n)

	return stateDict, nil
}

/*
CrossEntropyLoss computes softmax cross-entropy between logits (data[0]) and
one-hot targets (data[1]). Returns a single-element slice containing the loss.
*/
type CrossEntropyLoss struct{}

func NewCrossEntropyLoss() *CrossEntropyLoss { return &CrossEntropyLoss{} }

func (cel *CrossEntropyLoss) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("train.cross_entropy_loss", 2); err != nil {
		return nil, err
	}

	logits, targets := stateDict.Inputs[0], stateDict.Inputs[1]

	if len(targets) != len(logits) {
		return nil, fmt.Errorf("train.cross_entropy_loss: logits and target lengths differ")
	}

	stateDict.EnsureOperationOutLen(1)

	if len(logits) == 0 {
		stateDict.Out[0] = 0

		return stateDict, nil
	}

	probs := softmax(logits)
	loss := 0.0

	for index, target := range targets {
		loss -= math.Log(probs[index]+1e-9) * target
	}

	stateDict.Out[0] = loss

	return stateDict, nil
}

/*
MSEGrad computes the gradient of MSE loss with respect to predictions.
Inputs: data[0] = predictions, data[1] = targets.
*/
type MSEGrad struct{}

func NewMSEGrad() *MSEGrad { return &MSEGrad{} }

func (g *MSEGrad) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("train.mse_grad", 2); err != nil {
		return nil, err
	}

	predictions, targets := stateDict.Inputs[0], stateDict.Inputs[1]
	n := len(predictions)

	if len(targets) != n {
		return nil, fmt.Errorf("train.mse_grad: prediction and target lengths differ")
	}

	stateDict.EnsureOperationOutLen(n)

	if n == 0 {
		return stateDict, nil
	}

	scale := 2.0 / float64(n)
	copy(stateDict.Out, predictions)
	addScaledVec(stateDict.Out, targets, -1)
	scaleVec(stateDict.Out, scale)

	return stateDict, nil
}

/*
CrossEntropyGrad computes d(cross_entropy)/d(logits) = softmax(logits) - targets.
Inputs: data[0] = logits, data[1] = one-hot targets.
*/
type CrossEntropyGrad struct{}

func NewCrossEntropyGrad() *CrossEntropyGrad { return &CrossEntropyGrad{} }

func (g *CrossEntropyGrad) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("train.cross_entropy_grad", 2); err != nil {
		return nil, err
	}

	logits, targets := stateDict.Inputs[0], stateDict.Inputs[1]

	if len(targets) != len(logits) {
		return nil, fmt.Errorf("train.cross_entropy_grad: logits and target lengths differ")
	}

	stateDict.EnsureOperationOutLen(len(logits))

	if len(logits) == 0 {
		return stateDict, nil
	}

	probs := softmax(logits)
	addScaledVec(probs, targets, -1)
	copy(stateDict.Out, probs)

	return stateDict, nil
}

func softmax(xs []float64) []float64 {
	if len(xs) == 0 {
		return []float64{}
	}

	out := make([]float64, len(xs))
	maxValue := xs[0]

	for _, value := range xs[1:] {
		if value > maxValue {
			maxValue = value
		}
	}

	sum := 0.0

	for index, value := range xs {
		expValue := math.Exp(value - maxValue)
		out[index] = expValue
		sum += expValue
	}

	if sum == 0 {
		return out
	}

	scaleVec(out, 1.0/sum)
	return out
}

func addScaledVec(dst, src []float64, scale float64) {
	for index := range src {
		dst[index] += scale * src[index]
	}
}

func scaleVec(values []float64, scale float64) {
	for index := range values {
		values[index] *= scale
	}
}

func l2NormSq(values []float64) float64 {
	sum := 0.0

	for _, value := range values {
		sum += value * value
	}

	return sum
}
