//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "activation.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// CUDAActivation dispatches activation kernels to the GPU via CUDA.
// Device memory is managed internally by the C wrappers.
type CUDAActivation struct{}

// New creates a CUDAActivation.  No explicit initialization is required —
// CUDA context creation is lazy via the CUDA runtime.
func New() *CUDAActivation {
	return &CUDAActivation{}
}

// Forward dispatches to ReLU using the universal Operation signature.
//
// shape is optional metadata: when non-empty, the product of dimensions must equal len(data[0]);
// when empty, only len(data[0]) is used (caller must keep shape consistent elsewhere).
// Returns an error if data is missing or shape does not match the flattened input length.
func (c *CUDAActivation) Forward(shape []int, data ...[]float64) ([]float64, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("cuda activation: Forward requires at least one input slice")
	}

	input := data[0]

	if len(shape) > 0 {
		product := 1

		for _, dimension := range shape {
			if dimension < 0 {
				return nil, fmt.Errorf("cuda activation: negative shape dimension %d", dimension)
			}

			product *= dimension
		}

		if product != len(input) {
			return nil, fmt.Errorf(
				"cuda activation: shape product %d does not match len(data[0])=%d",
				product, len(input),
			)
		}
	}

	return c.ReLU(input)
}

// ReLU computes max(x, 0) element-wise.
func (c *CUDAActivation) ReLU(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.cuda_relu(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_relu failed (rc=%d)", rc)
	}
	return dst, nil
}

// LeakyReLU computes x if x >= 0 else alpha*x element-wise.
func (c *CUDAActivation) LeakyReLU(input []float64, alpha float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.cuda_leaky_relu(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.double(alpha),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_leaky_relu failed (rc=%d)", rc)
	}
	return dst, nil
}

// GELU computes the manifest GELU contract:
// 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
func (c *CUDAActivation) GELU(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.cuda_gelu(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_gelu failed (rc=%d)", rc)
	}
	return dst, nil
}

// Tanh computes element-wise hyperbolic tangent.
func (c *CUDAActivation) Tanh(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.cuda_tanh(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_tanh failed (rc=%d)", rc)
	}
	return dst, nil
}

// Sigmoid computes 1/(1+exp(-x)) element-wise.
func (c *CUDAActivation) Sigmoid(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.cuda_sigmoid(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_sigmoid failed (rc=%d)", rc)
	}
	return dst, nil
}

// Swish computes x/(1+exp(-x)) element-wise.
func (c *CUDAActivation) Swish(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.cuda_swish(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_swish failed (rc=%d)", rc)
	}
	return dst, nil
}

// SELU computes the self-normalizing scaled ELU element-wise.
func (c *CUDAActivation) SELU(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.cuda_selu(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_selu failed (rc=%d)", rc)
	}
	return dst, nil
}

// SwiGLU computes sigmoid(gate[i]) * value[i].
// input must have 2*n elements: first n are gates, second n are values.
// Returns n elements.
func (c *CUDAActivation) SwiGLU(input []float64) ([]float64, error) {
	if len(input)%2 != 0 {
		return nil, fmt.Errorf("swiglu: input length %d is not even", len(input))
	}
	n := len(input) / 2
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.cuda_swiglu(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_swiglu failed (rc=%d)", rc)
	}
	return dst, nil
}
