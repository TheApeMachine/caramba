//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "shape.h"
import "C"

import (
	"fmt"
	"strings"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
)

// MetalShapeOps dispatches shape manipulation kernels to the GPU via Metal.
type MetalShapeOps struct {
	metallib string
}

type ShapeForwardRequest struct {
	Op       string
	Metadata map[string]any
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

func (m *MetalShapeOps) Forward(
	request ShapeForwardRequest,
	shape []int,
	data ...[]float64,
) ([]float64, error) {
	node := executor.NodeSpec{Shape: shape, Metadata: request.Metadata}

	switch strings.ToLower(request.Op) {
	case "shape.reshape":
		if len(data) != 1 {
			return nil, fmt.Errorf("metal shape Forward reshape: requires one input buffer")
		}

		return m.Copy(data[0])
	case "shape.transpose":
		if len(data) != 1 {
			return nil, fmt.Errorf("metal shape Forward transpose: requires one input buffer")
		}

		return m.Transpose(
			shape,
			metalIntConfig(node, "dim0", 0),
			metalIntConfig(node, "dim1", 1),
			data[0],
		)
	case "shape.concat":
		if len(data) != 2 {
			return nil, fmt.Errorf("metal shape Forward concat: requires two input buffers")
		}

		return m.Concat(data[0], data[1])
	case "shape.view_as_heads":
		if len(data) != 1 {
			return nil, fmt.Errorf("metal shape Forward view_as_heads: requires one input buffer")
		}

		if len(shape) != 3 {
			return nil, fmt.Errorf("metal shape Forward view_as_heads: shape must be [B,T,D]")
		}

		numHeads := metalIntConfig(node, "num_heads", 1)
		if numHeads <= 0 || shape[2]%numHeads != 0 {
			return nil, fmt.Errorf("metal shape Forward view_as_heads: D must divide num_heads")
		}

		return m.ViewAsHeads(data[0], shape[0], shape[1], numHeads, shape[2]/numHeads)
	case "shape.merge_heads":
		if len(data) != 1 {
			return nil, fmt.Errorf("metal shape Forward merge_heads: requires one input buffer")
		}

		if len(shape) != 4 {
			return nil, fmt.Errorf("metal shape Forward merge_heads: shape must be [B,H,T,headDim]")
		}

		return m.MergeHeads(data[0], shape[0], shape[1], shape[2], shape[3])
	default:
		return nil, fmt.Errorf("metal shape Forward: unsupported operation %q", request.Op)
	}
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

// Split returns equal-sized chunks along a logical dimension as one flat buffer.
// outer is the product of dimensions before the split dimension, dimSize is the
// size of the logical dimension being split, splitSize is the number of elements
// per chunk along that dimension, and inner is the product of dimensions after
// the split dimension.
func (m *MetalShapeOps) Split(
	input []float64,
	outer int,
	dimSize int,
	splitSize int,
	inner int,
) ([]float64, error) {
	n := len(input)
	if n == 0 {
		return []float64{}, nil
	}

	if outer <= 0 || dimSize <= 0 || splitSize <= 0 || inner <= 0 {
		return nil, fmt.Errorf(
			"invalid split params: outer=%d dimSize=%d splitSize=%d inner=%d",
			outer, dimSize, splitSize, inner,
		)
	}

	if dimSize%splitSize != 0 {
		return nil, fmt.Errorf(
			"invalid split params: dimSize %d is not divisible by splitSize %d",
			dimSize, splitSize,
		)
	}

	expected := int64(outer) * int64(dimSize) * int64(inner)

	if expected != int64(n) {
		return nil, fmt.Errorf(
			"invalid split params: input length %d does not match outer*dimSize*inner=%d",
			n, expected,
		)
	}

	src32 := toFloat32(input)
	dst32 := make([]float32, n)
	rc := C.metal_split(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(outer), C.int(dimSize), C.int(splitSize), C.int(inner),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_split failed (rc=%d)", rc)
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

func metalIntConfig(node executor.NodeSpec, key string, fallback int) int {
	value, ok := node.Metadata[key]

	if !ok {
		return fallback
	}

	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case float32:
		return int(typed)
	default:
		return fallback
	}
}
