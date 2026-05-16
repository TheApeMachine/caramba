//go:build linux && cgo && cuda

package cuda

// #include <cuda_runtime.h>
import "C"

import "fmt"

func Available() error {
	var deviceCount C.int
	status := C.cudaGetDeviceCount(&deviceCount)

	if status != C.cudaSuccess {
		return fmt.Errorf("CUDA backend unavailable: cudaGetDeviceCount failed")
	}

	if deviceCount <= 0 {
		return fmt.Errorf("CUDA backend unavailable: no CUDA device visible")
	}

	return nil
}
