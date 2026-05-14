//go:build !darwin || !cgo

package metal

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

type Registry struct{}

func NewOptimizerRegistry() Registry { return Registry{} }

func unavailableOptimizer() (state.Optimizer, error) { return nil, metalUnavailable() }

func (registry Registry) Adam(*state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) AdamW(*state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) AdaMax(*state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) SGD(*state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) Lion(*state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) RMSProp(*state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) Hebbian(*state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) Lars(*state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) Lamb(*state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) AdaGrad(*state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) AdaDelta(*state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) LBFGS(*state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}
