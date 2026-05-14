//go:build cgo && xla

package xla

// XLA masking backend via the PJRT C API.
//
// Configure PJRT paths under compute.xla in cmd/asset/config.yml before runtime validation.

// #include <stdlib.h>
// #include "masking.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// XLAMasking dispatches masking operations to the XLA runtime via PJRT.
type XLAMasking struct {
	platform string
}

// NewMasking initialises the PJRT client for masking operations.
func NewMasking(platform string) (*XLAMasking, error) {
	config, err := NewPJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	if err := config.ValidateRuntime(); err != nil {
		return nil, err
	}

	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_masking_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_masking_init failed for platform %q", platform)
	}
	return &XLAMasking{platform: platform}, nil
}

// Shutdown releases all PJRT masking resources.
func (x *XLAMasking) Shutdown() { C.xla_masking_shutdown() }

// ---------------------------------------------------------------------------
// CausalMask
// ---------------------------------------------------------------------------

// XLACausalMask wraps XLAMasking for the CausalMask operation.
type XLACausalMask struct{ x *XLAMasking }

func (x *XLAMasking) NewCausalMask() *XLACausalMask {
	return &XLACausalMask{x: x}
}

// Forward generates a causal mask on XLA.
// shape must contain seq_len as its last element; data is unused.
func (op *XLACausalMask) Forward(shape []int, data ...[]float64) ([]float64, error) {
	seqLen := shape[len(shape)-1]
	return op.x.CausalMask(seqLen)
}

// CausalMask generates a causal attention mask.
func (x *XLAMasking) CausalMask(seqLen int) ([]float64, error) {
	n2 := seqLen * seqLen
	if n2 == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n2)
	rc := C.xla_causal_mask(
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(seqLen),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_causal_mask failed")
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// ApplyMask
// ---------------------------------------------------------------------------

// XLAApplyMask wraps XLAMasking for the ApplyMask operation.
type XLAApplyMask struct{ x *XLAMasking }

func (x *XLAMasking) NewApplyMask() *XLAApplyMask {
	return &XLAApplyMask{x: x}
}

// Forward applies mask to scores: out[i] = scores[i] + mask[i].
// data[0] = scores, data[1] = mask.
func (op *XLAApplyMask) Forward(shape []int, data ...[]float64) ([]float64, error) {
	return op.x.ApplyMask(data[0], data[1])
}

// ApplyMask computes scores + mask elementwise on XLA.
func (x *XLAMasking) ApplyMask(scores, mask []float64) ([]float64, error) {
	n := len(scores)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.xla_apply_mask(
		(*C.double)(unsafe.Pointer(&scores[0])),
		(*C.double)(unsafe.Pointer(&mask[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_apply_mask failed")
	}
	return dst, nil
}
