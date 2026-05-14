package compute

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

type Optimizer = state.Optimizer

/*
OptimizerRegistry exposes the curated first-order optimizers that every production
ComputeBackend ought to accelerate for its Location.

Implementations route to SIMD, accelerator kernels, or portable Go as appropriate; they MUST
reject kinds they cannot run without approximation instead of handing off silently to host
floating point with different rounding guarantees.
*/
type OptimizerRegistry interface {
	Adam(*state.Dict) (Optimizer, error)
	AdamW(*state.Dict) (Optimizer, error)
	AdaMax(*state.Dict) (Optimizer, error)
	SGD(*state.Dict) (Optimizer, error)
	Lion(*state.Dict) (Optimizer, error)
	RMSProp(*state.Dict) (Optimizer, error)
	Hebbian(*state.Dict) (Optimizer, error)
	Lars(*state.Dict) (Optimizer, error)
	Lamb(*state.Dict) (Optimizer, error)
	AdaGrad(*state.Dict) (Optimizer, error)
	AdaDelta(*state.Dict) (Optimizer, error)
	LBFGS(*state.Dict) (Optimizer, error)
}

func NewOptimizerRegistry(registryType OptimizerRegistry) OptimizerRegistry {
	return registryType
}
