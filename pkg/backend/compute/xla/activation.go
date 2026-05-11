//go:build cgo && xla

package xla

// XLA activation backend via the PJRT C API.
//
// Build requirements:
//   - XLA headers on the include path (set XLA_INCLUDE via CGO_CPPFLAGS)
//   - PJRT plugin shared library for your platform on LD_LIBRARY_PATH / DYLD_LIBRARY_PATH
//   - Compile activation_xla.cc alongside this package (CGo picks it up automatically)
//
// Example build:
//   CGO_CPPFLAGS="-I/path/to/xla" \
//   CGO_LDFLAGS="-ldl -lstdc++" \
//   go build -tags "cgo xla" ./backend/compute/xla/

// #cgo CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ldl -lstdc++
// #include <stdlib.h>
// #include "activation.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// XLAActivation dispatches activation functions to the XLA runtime via PJRT.
// Compiled executables are cached per element count; alpha-parameterised ops
// (leaky_relu) are recompiled per call because the alpha is baked into the
// StableHLO constant.
type XLAActivation struct {
	platform string
}

// New initialises the PJRT client for the given platform ("cpu" or "gpu").
func New(platform string) (*XLAActivation, error) {
	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_init failed for platform %q", platform)
	}
	return &XLAActivation{platform: platform}, nil
}

// Shutdown releases all PJRT resources.
func (x *XLAActivation) Shutdown() {
	C.xla_shutdown()
}

// ensureCompiled compiles executables for n if not already done.
func (x *XLAActivation) ensureCompiled(n int) error {
	if rc := C.xla_compile_activations(C.int(n)); rc != 0 {
		return fmt.Errorf("xla_compile_activations(%d) failed", n)
	}
	return nil
}

// Forward dispatches to ReLU with the new universal signature.
// shape is metadata only; data[0] is the primary input buffer.
func (x *XLAActivation) Forward(shape []int, data ...[]float64) []float64 {
	out, _ := x.ReLU(data[0])
	return out
}

// ReLU computes max(x, 0) element-wise.
func (x *XLAActivation) ReLU(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	if err := x.ensureCompiled(n); err != nil {
		return nil, err
	}
	dst := make([]float64, n)
	rc := C.xla_relu(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_relu failed")
	}
	return dst, nil
}

// LeakyReLU computes x if x>=0 else alpha*x.
// The executable is recompiled each call with the provided alpha baked in.
func (x *XLAActivation) LeakyReLU(input []float64, alpha float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.xla_leaky_relu(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.double(alpha),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_leaky_relu failed")
	}
	return dst, nil
}

// GELU computes the Gaussian Error Linear Unit (tanh approximation).
func (x *XLAActivation) GELU(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	if err := x.ensureCompiled(n); err != nil {
		return nil, err
	}
	dst := make([]float64, n)
	rc := C.xla_gelu(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_gelu failed")
	}
	return dst, nil
}

// Tanh computes element-wise hyperbolic tangent.
func (x *XLAActivation) Tanh(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	if err := x.ensureCompiled(n); err != nil {
		return nil, err
	}
	dst := make([]float64, n)
	rc := C.xla_tanh_act(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_tanh failed")
	}
	return dst, nil
}

// Sigmoid computes 1/(1+exp(-x)) element-wise via stablehlo.logistic.
func (x *XLAActivation) Sigmoid(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	if err := x.ensureCompiled(n); err != nil {
		return nil, err
	}
	dst := make([]float64, n)
	rc := C.xla_sigmoid(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_sigmoid failed")
	}
	return dst, nil
}

// SwiGLU computes sigmoid(gate[i]) * value[i].
// input must have 2*n elements (gates first, then values); output is n elements.
func (x *XLAActivation) SwiGLU(input []float64) ([]float64, error) {
	if len(input)%2 != 0 {
		return nil, fmt.Errorf("swiglu: input length %d is not even", len(input))
	}
	n := len(input) / 2
	if n == 0 {
		return []float64{}, nil
	}
	if err := x.ensureCompiled(n); err != nil {
		return nil, err
	}
	dst := make([]float64, n)
	rc := C.xla_swiglu(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_swiglu failed")
	}
	return dst, nil
}
