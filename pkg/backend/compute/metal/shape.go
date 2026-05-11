//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "shape.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// MetalShapeOps dispatches shape manipulation kernels to the GPU via Metal.
type MetalShapeOps struct {
	metallib string
}

// NewShapeOps creates and initializes a MetalShapeOps.
// metallib must be the absolute path to shape.metallib compiled from shape.metal.
func NewShapeOps(metallib string) (*MetalShapeOps, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))
	if rc := C.metal_shape_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_shape_init failed (rc=%d): check %q exists", rc, metallib)
	}
	return &MetalShapeOps{metallib: metallib}, nil
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------

// Transpose swaps dim0 and dim1 of the tensor described by shape.
// data[0] is the flat input buffer; returns the transposed flat buffer.
func (m *MetalShapeOps) Transpose(shape []int, dim0, dim1 int, data []float64) ([]float64, error) {
	n := len(data)
	if n == 0 {
		return []float64{}, nil
	}
	src32 := toFloat32(data)
	dst32 := make([]float32, n)

	rank := len(shape)
	cshape := make([]C.int, rank)
	for i, v := range shape {
		cshape[i] = C.int(v)
	}

	rc := C.metal_transpose(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		(*C.int)(unsafe.Pointer(&cshape[0])),
		C.int(rank),
		C.int(dim0),
		C.int(dim1),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_transpose failed (rc=%d)", rc)
	}
	return toFloat64(dst32), nil
}

// Forward dispatches to Transpose using shape metadata from the shape slice.
// Expects: data[0]=input, shape=tensor shape; Dim0 and Dim1 encoded at
// positions shape[len(shape)-2] and shape[len(shape)-1] is NOT the pattern —
// use the Transpose method directly for full control.
// This satisfies the universal Forward interface as a no-op copy fallback.
func (m *MetalShapeOps) Forward(shape []int, data ...[]float64) []float64 {
	out, err := m.Copy(data[0])
	if err != nil {
		panic(err)
	}

	return out
}

// ---------------------------------------------------------------------------
// Copy (Reshape)
// ---------------------------------------------------------------------------

// Copy copies the input buffer unchanged (reshape — only logical shape differs).
func (m *MetalShapeOps) Copy(input []float64) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	src32 := toFloat32(input)
	dst32 := make([]float32, n)
	rc := C.metal_copy(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(n),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_copy failed (rc=%d)", rc)
	}
	return toFloat64(dst32), nil
}

// ---------------------------------------------------------------------------
// Concat (two-tensor version; call repeatedly for more tensors)
// ---------------------------------------------------------------------------

// Concat concatenates two flat buffers.
func (m *MetalShapeOps) Concat(a, b []float64) ([]float64, error) {
	na, nb := len(a), len(b)
	if na == 0 && nb == 0 {
		return []float64{}, nil
	}
	a32 := toFloat32(a)
	b32 := toFloat32(b)
	dst32 := make([]float32, na+nb)

	var pA, pB *C.float
	if na > 0 {
		pA = (*C.float)(unsafe.Pointer(&a32[0]))
	}
	if nb > 0 {
		pB = (*C.float)(unsafe.Pointer(&b32[0]))
	}

	rc := C.metal_concat(pA, C.int(na), pB, C.int(nb),
		(*C.float)(unsafe.Pointer(&dst32[0])))
	if rc != 0 {
		return nil, fmt.Errorf("metal_concat failed (rc=%d)", rc)
	}
	return toFloat64(dst32), nil
}

// ---------------------------------------------------------------------------
// ViewAsHeads: [B,T,D] -> [B,H,T,D/H]
// ---------------------------------------------------------------------------

// ViewAsHeads converts [B,T,D] to [B,H,T,D/H].
// The input is first logically reshaped to [B,T,H,head_dim] then transposed.
func (m *MetalShapeOps) ViewAsHeads(input []float64, B, T, H, headDim int) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	src32 := toFloat32(input)
	dst32 := make([]float32, n)
	rc := C.metal_view_as_heads(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(B), C.int(T), C.int(H), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_view_as_heads failed (rc=%d)", rc)
	}
	return toFloat64(dst32), nil
}

// ---------------------------------------------------------------------------
// MergeHeads: [B,H,T,head_dim] -> [B,T,H*head_dim]
// ---------------------------------------------------------------------------

// MergeHeads converts [B,H,T,head_dim] to [B,T,H*head_dim].
func (m *MetalShapeOps) MergeHeads(input []float64, B, H, T, headDim int) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}
	src32 := toFloat32(input)
	dst32 := make([]float32, n)
	rc := C.metal_merge_heads(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(B), C.int(H), C.int(T), C.int(headDim),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_merge_heads failed (rc=%d)", rc)
	}
	return toFloat64(dst32), nil
}
