//go:build cgo && xla

package xla

// XLA masking backend via the PJRT C API.
//
// Build requirements (same as activation.go):
//   CGO_CPPFLAGS="-I/path/to/xla" \
//   go build -tags "cgo xla" ./pk./pkg/backend/compute/xla

// #include <stdlib.h>
// #include <stdlib.h>
// #include "masking.h"
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// XLAMasking dispatches masking operations to the XLA runtime via PJRT.
type XLAMasking struct {
	platform string
}

// NewMasking initialises the PJRT client for masking operations.
func NewMasking(platform string) (*XLAMasking, error) {
	if err := NewPJRTConfig(platform).ValidateRuntime(); err != nil {
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
func (op *XLACausalMask) Forward(shape []int, data ...[]float64) []float64 {
	seqLen := shape[len(shape)-1]
	out, err := op.x.CausalMask(seqLen)
	if err != nil {
		return causalMaskScalarXLA(seqLen)
	}
	return out
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

func causalMaskScalarXLA(seqLen int) []float64 {
	ninf := math.Inf(-1)
	out := make([]float64, seqLen*seqLen)
	for i := 0; i < seqLen; i++ {
		base := i * seqLen
		for j := 0; j <= i; j++ {
			out[base+j] = 0.0
		}
		for j := i + 1; j < seqLen; j++ {
			out[base+j] = ninf
		}
	}
	return out
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
func (op *XLAApplyMask) Forward(shape []int, data ...[]float64) []float64 {
	out, err := op.x.ApplyMask(data[0], data[1])
	if err != nil {
		res := make([]float64, len(data[0]))
		for i := range data[0] {
			res[i] = data[0][i] + data[1][i]
		}
		return res
	}
	return out
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
