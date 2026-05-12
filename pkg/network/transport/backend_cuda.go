//go:build linux && cgo && cuda

package transport

import computecuda "github.com/theapemachine/caramba/pkg/backend/compute/cuda"

func init() {
	acceleratorStreamBackendFactories = append(
		acceleratorStreamBackendFactories,
		registeredStreamBackendFactory{
			name:    "cuda",
			factory: NewCUDAStreamBackend,
		},
	)
}

func NewCUDAStreamBackend() (StreamComputeBackend, error) {
	return computecuda.NewTensorBackend(), nil
}
