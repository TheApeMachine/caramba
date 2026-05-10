//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "masking.h"
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// CUDAMasking dispatches masking kernels to the GPU via CUDA.
type CUDAMasking struct{}

// NewMasking creates a CUDAMasking.
func NewMasking() *CUDAMasking { return &CUDAMasking{} }

// ---------------------------------------------------------------------------
// CausalMask
// ---------------------------------------------------------------------------

// CUDACausalMask wraps CUDAMasking for the CausalMask operation.
type CUDACausalMask struct{ m *CUDAMasking }

func (c *CUDAMasking) NewCausalMask() *CUDACausalMask {
	return &CUDACausalMask{m: c}
}

// Forward generates a causal mask on the GPU.
// shape must contain seq_len as its last element; data is unused.
// Returns []float64 of length seq_len*seq_len.
func (op *CUDACausalMask) Forward(shape []int, data ...[]float64) []float64 {
	seqLen := shape[len(shape)-1]
	out, err := op.m.CausalMask(seqLen)
	if err != nil {
		// Fallback to scalar Go
		return causalMaskScalarCUDA(seqLen)
	}
	return out
}

// CausalMask generates a causal attention mask of size seq_len x seq_len.
func (c *CUDAMasking) CausalMask(seqLen int) ([]float64, error) {
	n := seqLen * seqLen
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.cuda_causal_mask(
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(seqLen),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_causal_mask failed (rc=%d)", rc)
	}
	return dst, nil
}

func causalMaskScalarCUDA(seqLen int) []float64 {
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

// CUDAApplyMask wraps CUDAMasking for the ApplyMask operation.
type CUDAApplyMask struct{ m *CUDAMasking }

func (c *CUDAMasking) NewApplyMask() *CUDAApplyMask {
	return &CUDAApplyMask{m: c}
}

// Forward applies mask to scores: out[i] = scores[i] + mask[i].
// data[0] = scores, data[1] = mask.
func (op *CUDAApplyMask) Forward(shape []int, data ...[]float64) []float64 {
	out, err := op.m.ApplyMask(data[0], data[1])
	if err != nil {
		// Fallback
		res := make([]float64, len(data[0]))
		for i := range data[0] {
			res[i] = data[0][i] + data[1][i]
		}
		return res
	}
	return out
}

// ApplyMask computes scores + mask elementwise on the GPU.
func (c *CUDAMasking) ApplyMask(scores, mask []float64) ([]float64, error) {
	n := len(scores)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.cuda_apply_mask(
		(*C.double)(unsafe.Pointer(&scores[0])),
		(*C.double)(unsafe.Pointer(&mask[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_apply_mask failed (rc=%d)", rc)
	}
	return dst, nil
}
