package transport

import computecpu "github.com/theapemachine/caramba/pkg/backend/compute/cpu"

var acceleratorStreamBackendFactories []registeredStreamBackendFactory

type registeredStreamBackendFactory struct {
	name    string
	factory StreamBackendFactory
}

func NewCPUStreamBackend() (StreamComputeBackend, error) {
	return computecpu.NewTensorBackend(), nil
}
