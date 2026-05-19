// Package causal implements causal-inference primitives on the CPU.
package causal

import "github.com/theapemachine/caramba/pkg/backend/compute/kernels"

type (
	Kernel    = kernels.Kernel
	Signature = kernels.Signature
)

var Default = kernels.Default
