//go:build !cgo || !xla

package xla

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

const xlaOptimizerUnavailableMsg = "xla optimizer: unavailable without cgo and xla build tags"

type Registry struct{}

func NewOptimizerRegistry() Registry {
	return Registry{}
}

func (registry Registry) Adam(*state.Dict) (state.Optimizer, error) {
	return nil, xlaOptimizerUnavailable()
}

func (registry Registry) AdamW(*state.Dict) (state.Optimizer, error) {
	return nil, xlaOptimizerUnavailable()
}

func (registry Registry) AdaMax(*state.Dict) (state.Optimizer, error) {
	return nil, xlaOptimizerUnavailable()
}

func (registry Registry) SGD(*state.Dict) (state.Optimizer, error) {
	return nil, xlaOptimizerUnavailable()
}

func (registry Registry) Lion(*state.Dict) (state.Optimizer, error) {
	return nil, xlaOptimizerUnavailable()
}

func (registry Registry) RMSProp(*state.Dict) (state.Optimizer, error) {
	return nil, xlaOptimizerUnavailable()
}

func (registry Registry) Hebbian(*state.Dict) (state.Optimizer, error) {
	return nil, xlaOptimizerUnavailable()
}

func (registry Registry) Lars(*state.Dict) (state.Optimizer, error) {
	return nil, xlaOptimizerUnavailable()
}

func (registry Registry) Lamb(*state.Dict) (state.Optimizer, error) {
	return nil, xlaOptimizerUnavailable()
}

func (registry Registry) AdaGrad(*state.Dict) (state.Optimizer, error) {
	return nil, xlaOptimizerUnavailable()
}

func (registry Registry) AdaDelta(*state.Dict) (state.Optimizer, error) {
	return nil, xlaOptimizerUnavailable()
}

func (registry Registry) LBFGS(*state.Dict) (state.Optimizer, error) {
	return nil, xlaOptimizerUnavailable()
}

func xlaOptimizerUnavailable() error {
	return fmt.Errorf("%s", xlaOptimizerUnavailableMsg)
}
