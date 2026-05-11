//go:build cgo && xla

package xla

// XLA shape backend via the PJRT C API.
//
// Build requirements:
//   - Same as activation.go (XLA headers, PJRT plugin library).
//   - xla_init() must be called before using any XLAShapeOps method.

// #include "shape.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// XLAShapeOps dispatches shape manipulation operations to the XLA runtime.
type XLAShapeOps struct{}

// NewShapeOps creates an XLAShapeOps.
// xla_init (via XLAActivation.New) must already have been called.
func NewShapeOps() *XLAShapeOps {
	return &XLAShapeOps{}
}

// Forward satisfies the universal operation interface — performs a copy.
func (x *XLAShapeOps) Forward(shape []int, data ...[]float64) []float64 {
	out, err := x.Copy(data[0])
	if err != nil {
		panic(err)
	}

	return out
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------

// Transpose swaps dim0 and dim1 of the tensor described by shape.
func (x *XLAShapeOps) Transpose(shape []int, dim0, dim1 int, input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rank := len(shape)
	cshape := make([]C.int, rank)
	for i, v := range shape {
		cshape[i] = C.int(v)
	}
	rc := C.xla_transpose(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.int)(unsafe.Pointer(&cshape[0])),
		C.int(rank),
		C.int(dim0),
		C.int(dim1),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_transpose failed")
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// Copy (Reshape)
// ---------------------------------------------------------------------------

// Copy copies the input buffer unchanged (reshape — logical shape differs).
func (x *XLAShapeOps) Copy(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.xla_copy(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_copy failed")
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// Concat
// ---------------------------------------------------------------------------

// Concat concatenates two flat buffers.
func (x *XLAShapeOps) Concat(a, b []float64) ([]float64, error) {
	na, nb := len(a), len(b)
	if na == 0 && nb == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, na+nb)

	var pA, pB *C.double
	if na > 0 {
		pA = (*C.double)(unsafe.Pointer(&a[0]))
	}
	if nb > 0 {
		pB = (*C.double)(unsafe.Pointer(&b[0]))
	}

	rc := C.xla_concat(pA, C.int(na), pB, C.int(nb),
		(*C.double)(unsafe.Pointer(&dst[0])))
	if rc != 0 {
		return nil, fmt.Errorf("xla_concat failed")
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// ViewAsHeads: [B,T,D] -> [B,H,T,D/H]
// ---------------------------------------------------------------------------

// ViewAsHeads converts [B,T,D] to [B,H,T,D/H].
func (x *XLAShapeOps) ViewAsHeads(input []float64, B, T, H, headDim int) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.xla_view_as_heads(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(B), C.int(T), C.int(H), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_view_as_heads failed")
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// MergeHeads: [B,H,T,head_dim] -> [B,T,H*head_dim]
// ---------------------------------------------------------------------------

// MergeHeads converts [B,H,T,headDim] to [B,T,H*headDim].
func (x *XLAShapeOps) MergeHeads(input []float64, B, H, T, headDim int) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.xla_merge_heads(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(B), C.int(H), C.int(T), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_merge_heads failed")
	}
	return dst, nil
}
