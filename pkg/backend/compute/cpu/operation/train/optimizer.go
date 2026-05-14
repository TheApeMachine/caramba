package train

import (
	cpuopt "github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/adam"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/lion"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/rmsprop"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/sgd"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
OptimizerStep wraps any Optimizer implementation as a graph node.
Inputs: data[0] = params, data[1] = grads.
Output: updated params.
*/
type OptimizerStep struct {
	inner cpuopt.Optimizer
	state *state.Dict
}

func newOptimizerStep(inner cpuopt.Optimizer, stateDict *state.Dict) *OptimizerStep {
	return &OptimizerStep{
		inner: inner,
		state: stateDict,
	}
}

func (optimizerStep *OptimizerStep) Forward(_ []int, data ...[]float64) []float64 {
	optimizerStep.state.
		WithParams(data[0]).
		WithGrads(data[1])

	updated, err := optimizerStep.inner.Step(optimizerStep.state)

	if err != nil {
		panic(err)
	}

	return updated.Out
}

/*
NewAdamStep creates an Adam optimizer node.
*/
func NewAdamStep(lr, beta1, beta2, eps, wd float64) *OptimizerStep {
	stateDict := optimizerState(lr, beta1, beta2, eps, wd)

	return newOptimizerStep(adam.NewAdam(), stateDict)
}

/*
NewAdamWStep creates an AdamW optimizer node (Adam + decoupled weight decay).
*/
func NewAdamWStep(lr, beta1, beta2, eps, wd float64) *OptimizerStep {
	stateDict := optimizerState(lr, beta1, beta2, eps, wd)

	return newOptimizerStep(adam.NewAdamW(), stateDict)
}

/*
NewSGDStep creates an SGD optimizer node.
*/
func NewSGDStep(lr, momentum, wd float64, nesterov bool) *OptimizerStep {
	stateDict := state.NewDict().
		WithLR(lr).
		WithMomentum(momentum).
		WithWD(wd).
		WithNesterov(nesterov)

	return newOptimizerStep(sgd.NewSGD(), stateDict)
}

/*
NewLionStep creates a Lion optimizer node.
*/
func NewLionStep(lr, beta1, beta2, wd float64) *OptimizerStep {
	stateDict := optimizerState(lr, beta1, beta2, 0, wd)

	return newOptimizerStep(lion.NewLion(), stateDict)
}

/*
NewRMSPropStep creates an RMSProp optimizer node.
*/
func NewRMSPropStep(lr, alpha, eps, momentum, wd float64) *OptimizerStep {
	stateDict := state.NewDict().
		WithLR(lr).
		WithAlpha(alpha).
		WithEps(eps).
		WithMomentum(momentum).
		WithWD(wd)

	return newOptimizerStep(rmsprop.NewRMSProp(), stateDict)
}

func optimizerState(lr, beta1, beta2, eps, wd float64) *state.Dict {
	return state.NewDict().
		WithLR(lr).
		WithBeta1(beta1).
		WithBeta2(beta2).
		WithEps(eps).
		WithWD(wd)
}
