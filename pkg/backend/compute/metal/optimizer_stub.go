//go:build !darwin || !cgo

package metal

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

type Registry struct{}

func NewOptimizerRegistry() Registry { return Registry{} }

func unavailableOptimizer() (state.Optimizer, error) { return nil, metalUnavailable() }

func (registry Registry) Adam(_ *state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) AdamW(_ *state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) AdaMax(_ *state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) SGD(_ *state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) Lion(_ *state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) RMSProp(_ *state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) Hebbian(_ *state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) Lars(_ *state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) Lamb(_ *state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) AdaGrad(_ *state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) AdaDelta(_ *state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}

func (registry Registry) LBFGS(_ *state.Dict) (state.Optimizer, error) {
	return unavailableOptimizer()
}
