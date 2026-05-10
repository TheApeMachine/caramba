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
	return &MetalMasking{metallib: metallib}, nil
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

// Forward generates a causal mask.
// shape must contain seq_len as its last element.
// data is unused.
// Returns a flat []float64 of length seq_len*seq_len.
func (op *MetalCausalMask) Forward(shape []int, data ...[]float64) []float64 {
	seqLen := shape[len(shape)-1]
	n := seqLen * seqLen
	dst32 := make([]float32, n)

	rc := C.metal_causal_mask(
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(seqLen),
	)
	if rc != 0 {
		// Fallback: scalar Go implementation
		return causalMaskScalarGo(seqLen)
	}
	return toFloat64(dst32)
}

// causalMaskScalarGo is the pure-Go fallback for Metal failures.
func causalMaskScalarGo(seqLen int) []float64 {
	const negInf = -3.4028234663852886e+38 // -FLT_MAX as float64
	out := make([]float64, seqLen*seqLen)
	for i := 0; i < seqLen; i++ {
		base := i * seqLen
		for j := 0; j <= i; j++ {
			out[base+j] = 0.0
		}
		for j := i + 1; j < seqLen; j++ {
			out[base+j] = negInf
		}
	}
	return out
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

// Forward computes scores + mask elementwise.
// data[0] = scores, data[1] = mask
func (op *MetalApplyMask) Forward(shape []int, data ...[]float64) []float64 {
	scores := data[0]
	mask := data[1]
	n := len(scores)
	if n == 0 {
		return []float64{}
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
		// Fallback
		out := make([]float64, n)
		for i := range scores {
			out[i] = scores[i] + mask[i]
		}
		return out
	}
	return toFloat64(dst32)
}
