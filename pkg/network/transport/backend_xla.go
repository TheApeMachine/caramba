//go:build cgo && xla

package transport

import computexla "github.com/theapemachine/caramba/pkg/backend/compute/xla"

func init() {
	acceleratorStreamBackendFactories = append(
		acceleratorStreamBackendFactories,
		registeredStreamBackendFactory{
			name:    "xla",
			factory: NewXLAStreamBackend,
		},
	)
}

func NewXLAStreamBackend() (StreamComputeBackend, error) {
	return computexla.NewTensorBackend("gpu")
}
