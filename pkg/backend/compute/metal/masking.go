//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "masking.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// MetalMasking holds the path to the compiled masking.metallib.
type MetalMasking struct {
	metallib string
	runtime  *MetalRuntime
}

// NewMasking creates and initialises a MetalMasking.
// metallib must be the absolute path to masking.metallib compiled from
// masking.metal:
//
//	xcrun -sdk macosx metal -c masking.metal -o masking.air
//	xcrun -sdk macosx metallib masking.air -o masking.metallib
func NewMasking(metallib string) (*MetalMasking, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_masking_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_masking_init failed (rc=%d): check that %q exists and Metal is available", rc, metallib)
	}

	runtime, err := newStandaloneMetalRuntime()
	if err != nil {
		return nil, err
	}

	return &MetalMasking{metallib: metallib, runtime: runtime}, nil
}

// ---------------------------------------------------------------------------
// CausalMask
// ---------------------------------------------------------------------------

// MetalCausalMask dispatches causal mask generation to Metal.
type MetalCausalMask struct {
	m *MetalMasking
}

func (m *MetalMasking) NewCausalMask() *MetalCausalMask {
	return &MetalCausalMask{m: m}
}

// Forward generates a causal mask on Metal and reports kernel failures.
func (op *MetalCausalMask) Forward(shape []int, data ...[]float64) ([]float64, error) {
	seqLen := shape[len(shape)-1]
	n := seqLen * seqLen

	if n == 0 {
		return []float64{}, nil
	}

	dst32 := make([]float32, n)

	rc := C.metal_causal_mask(
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(seqLen),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_causal_mask failed (rc=%d)", rc)
	}
	return toFloat64(dst32), nil
}

// ---------------------------------------------------------------------------
// ApplyMask
// ---------------------------------------------------------------------------

// MetalApplyMask dispatches additive mask application to Metal.
type MetalApplyMask struct {
	m *MetalMasking
}

func (m *MetalMasking) NewApplyMask() *MetalApplyMask {
	return &MetalApplyMask{m: m}
}

// Forward computes scores + mask elementwise on Metal and reports kernel failures.
func (op *MetalApplyMask) Forward(shape []int, data ...[]float64) ([]float64, error) {
	scores := data[0]
	mask := data[1]
	n := len(scores)
	if n == 0 {
		return []float64{}, nil
	}

	src32 := toFloat32(scores)
	msk32 := toFloat32(mask)
	dst32 := make([]float32, n)

	rc := C.metal_apply_mask(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&msk32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_apply_mask failed (rc=%d)", rc)
	}
	return toFloat64(dst32), nil
}
