package runtime

import (
	stdruntime "runtime"

	"github.com/theapemachine/caramba/pkg/backend/compute"
)

func backendType(backendName string) (compute.BackendType, error) {
	switch backendName {
	case "auto":
		return automaticBackendType(), nil
	case "cpu", "host":
		return compute.CPU, nil
	case "metal":
		return compute.METAL, nil
	case "cuda":
		return compute.CUDA, nil
	case "xla":
		return compute.XLA, nil
	}

	return compute.CPU, &unsupportedBackendError{name: backendName}
}

func automaticBackendType() compute.BackendType {
	switch stdruntime.GOOS {
	case "darwin":
		return compute.METAL
	case "linux":
		return compute.CUDA
	default:
		return compute.CPU
	}
}

type unsupportedBackendError struct {
	name string
}

func (err *unsupportedBackendError) Error() string {
	return "unsupported backend " + err.name
}
