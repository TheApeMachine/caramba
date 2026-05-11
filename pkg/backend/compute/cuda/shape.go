//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "shape.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// CUDAShapeOps dispatches shape manipulation kernels to the GPU via CUDA.
type CUDAShapeOps struct{}

// NewShapeOps creates a CUDAShapeOps.
func NewShapeOps() *CUDAShapeOps {
	return &CUDAShapeOps{}
}

// Forward satisfies the universal operation interface.
// It performs a no-op copy (reshape). Use the typed methods for full control.
// shape is unused (element count comes from data[0]).
func (c *CUDAShapeOps) Forward(_ []int, data ...[]float64) ([]float64, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("cuda shape: Forward requires data[0]")
	}

	return c.Copy(data[0])
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------

// Transpose swaps dim0 and dim1 of the tensor described by shape.
func (c *CUDAShapeOps) Transpose(shape []int, dim0, dim1 int, input []float64) ([]float64, error) {
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

	rc := C.cuda_transpose(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		(*C.int)(unsafe.Pointer(&cshape[0])),
		C.int(rank),
		C.int(dim0),
		C.int(dim1),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_transpose failed (rc=%d)", rc)
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// Copy (Reshape)
// ---------------------------------------------------------------------------

// Copy copies the input buffer unchanged (reshape).
func (c *CUDAShapeOps) Copy(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.cuda_copy(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_copy failed (rc=%d)", rc)
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// Concat
// ---------------------------------------------------------------------------

// Concat concatenates two flat buffers.
func (c *CUDAShapeOps) Concat(a, b []float64) ([]float64, error) {
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

	rc := C.cuda_concat(pA, C.int(na), pB, C.int(nb),
		(*C.double)(unsafe.Pointer(&dst[0])))
	if rc != 0 {
		return nil, fmt.Errorf("cuda_concat failed (rc=%d)", rc)
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// ViewAsHeads: [B,T,D] -> [B,H,T,D/H]
// ---------------------------------------------------------------------------

// ViewAsHeads converts [B,T,D] to [B,H,T,D/H].
// Input is first logically reshaped to [B,T,H,headDim] then transposed.
func (c *CUDAShapeOps) ViewAsHeads(input []float64, B, T, H, headDim int) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.cuda_view_as_heads(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(B), C.int(T), C.int(H), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_view_as_heads failed (rc=%d)", rc)
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// MergeHeads: [B,H,T,head_dim] -> [B,T,H*head_dim]
// ---------------------------------------------------------------------------

// MergeHeads converts [B,H,T,headDim] to [B,T,H*headDim].
func (c *CUDAShapeOps) MergeHeads(input []float64, B, H, T, headDim int) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	dst := make([]float64, n)
	rc := C.cuda_merge_heads(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(B), C.int(H), C.int(T), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_merge_heads failed (rc=%d)", rc)
	}
	return dst, nil
}
