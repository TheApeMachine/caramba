/*
Package kernels re-exports the neon kernel registry for callers that import
github.com/theapemachine/caramba/pkg/backend/compute/kernels.
*/
package kernels

import "github.com/theapemachine/caramba/pkg/backend/compute/kernels/neon"

type (
	Kernel    = neon.Kernel
	Signature = neon.Signature
	Registry  = neon.Registry
)

var Default = neon.Default

func NewRegistry() *Registry {
	return neon.NewRegistry()
}

var (
	DefaultDropoutConfig  = neon.DefaultDropoutConfig
	DefaultSamplingConfig = neon.DefaultSamplingConfig
)
