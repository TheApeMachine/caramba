//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "activation.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// MetalActivation holds the path to the compiled .metallib and ensures
// Metal is initialized before any dispatch.
type MetalActivation struct {
	metallib string
}

// New creates and initializes a MetalActivation.
// metallib must be the absolute path to activation.metallib compiled from
// activation.metal via:
//
//	xcrun -sdk macosx metal -c activation.metal -o activation.air
//	xcrun -sdk macosx metallib activation.air -o activation.metallib
func New(metallib string) (*MetalActivation, error) {
	cpath := C.CString(metallib)
	// C.CString allocates; free after call since metal_init copies internally.
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_init failed (rc=%d): check that %q exists and Metal is available", rc, metallib)
	}
	return &MetalActivation{metallib: metallib}, nil
}

// ---------------------------------------------------------------------------
// Helpers: float64 ↔ float32 conversion
// ---------------------------------------------------------------------------

func toFloat32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func toFloat64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

// ---------------------------------------------------------------------------
// Activation operations
// ---------------------------------------------------------------------------

// Forward dispatches to ReLU with the new universal signature.
// shape is metadata only; data[0] is the primary input buffer.
func (m *MetalActivation) Forward(shape []int, data ...[]float64) []float64 {
	out, err := m.ReLU(data[0])
	if err != nil {
		panic(err)
	}

	return out
}

// ReLU computes max(x, 0) element-wise.
func (m *MetalActivation) ReLU(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	src := toFloat32(input)
	dst := make([]float32, n)

	rc := C.metal_relu(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_relu failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// LeakyReLU computes x if x >= 0 else alpha*x element-wise.
func (m *MetalActivation) LeakyReLU(input []float64, alpha float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	src := toFloat32(input)
	dst := make([]float32, n)

	rc := C.metal_leaky_relu(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.float(alpha),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_leaky_relu failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// GELU computes the Gaussian Error Linear Unit (tanh approximation).
func (m *MetalActivation) GELU(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	src := toFloat32(input)
	dst := make([]float32, n)

	rc := C.metal_gelu(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_gelu failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// Tanh computes element-wise hyperbolic tangent (rational approximation).
func (m *MetalActivation) Tanh(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	src := toFloat32(input)
	dst := make([]float32, n)

	rc := C.metal_tanh(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_tanh failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// Sigmoid computes 1/(1+exp(-x)) element-wise.
func (m *MetalActivation) Sigmoid(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	src := toFloat32(input)
	dst := make([]float32, n)

	rc := C.metal_sigmoid(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_sigmoid failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}

// SwiGLU computes sigmoid(gate[i]) * value[i].
// input must have 2*n elements: the first n are gates, the second n are values.
// Returns n elements.
func (m *MetalActivation) SwiGLU(input []float64) ([]float64, error) {
	if len(input)%2 != 0 {
		return nil, fmt.Errorf("swiglu: input length %d is not even", len(input))
	}
	n := len(input) / 2
	if n == 0 {
		return []float64{}, nil
	}
	src := toFloat32(input) // length 2*n
	dst := make([]float32, n)

	rc := C.metal_swiglu(
		(*C.float)(unsafe.Pointer(&src[0])),
		(*C.float)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_swiglu failed (rc=%d)", rc)
	}
	return toFloat64(dst), nil
}
