//go:build !linux || !cgo || !cuda

package cuda

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

const cudaOptimizerUnavailableMsg = "CUDA optimizer unavailable: rebuild on Linux with CGO enabled and build tags linux,cgo,cuda"

type Registry struct{}

func NewOptimizerRegistry() Registry {
	return Registry{}
}

func (registry Registry) Adam(*state.Dict) (state.Optimizer, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry Registry) AdamW(*state.Dict) (state.Optimizer, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry Registry) AdaMax(*state.Dict) (state.Optimizer, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry Registry) SGD(*state.Dict) (state.Optimizer, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry Registry) Lion(*state.Dict) (state.Optimizer, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry Registry) RMSProp(*state.Dict) (state.Optimizer, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry Registry) Hebbian(*state.Dict) (state.Optimizer, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry Registry) Lars(*state.Dict) (state.Optimizer, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry Registry) Lamb(*state.Dict) (state.Optimizer, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry Registry) AdaGrad(*state.Dict) (state.Optimizer, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry Registry) AdaDelta(*state.Dict) (state.Optimizer, error) {
	return nil, cudaOptimizerUnavailable()
}

func (registry Registry) LBFGS(*state.Dict) (state.Optimizer, error) {
	return nil, cudaOptimizerUnavailable()
}

func cudaOptimizerUnavailable() error {
	return fmt.Errorf("%s", cudaOptimizerUnavailableMsg)
}
