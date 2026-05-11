package train

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/adam"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/sgd"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/lion"
	"github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer/rmsprop"
	cpuopt "github.com/theapemachine/caramba/pkg/backend/compute/cpu/optimizer"
)

/*
OptimizerStep wraps any Optimizer implementation as a graph node.
Inputs: data[0] = params, data[1] = grads.
Output: updated params.
*/
type OptimizerStep struct {
	inner cpuopt.Optimizer
}

func newOptimizerStep(inner cpuopt.Optimizer) *OptimizerStep {
	return &OptimizerStep{inner: inner}
}

func (os *OptimizerStep) Forward(_ []int, data ...[]float64) []float64 {
	return os.inner.Step(data[0], data[1])
}

/*
NewAdamStep creates an Adam optimizer node.
*/
func NewAdamStep(lr, beta1, beta2, eps, wd float64) *OptimizerStep {
	return newOptimizerStep(adam.NewAdam(lr, beta1, beta2, eps, wd))
}

/*
NewAdamWStep creates an AdamW optimizer node (Adam + decoupled weight decay).
*/
func NewAdamWStep(lr, beta1, beta2, eps, wd float64) *OptimizerStep {
	return newOptimizerStep(adam.NewAdamW(lr, beta1, beta2, eps, wd))
}

/*
NewSGDStep creates an SGD optimizer node.
*/
func NewSGDStep(lr, momentum, wd float64, nesterov bool) *OptimizerStep {
	return newOptimizerStep(sgd.NewSGD(lr, momentum, wd, nesterov))
}

/*
NewLionStep creates a Lion optimizer node.
*/
func NewLionStep(lr, beta1, beta2, wd float64) *OptimizerStep {
	return newOptimizerStep(lion.NewLion(lr, beta1, beta2, wd))
}

/*
NewRMSPropStep creates an RMSProp optimizer node.
*/
func NewRMSPropStep(lr, alpha, eps, momentum, wd float64) *OptimizerStep {
	return newOptimizerStep(rmsprop.NewRMSProp(lr, alpha, eps, momentum, wd, false))
}
