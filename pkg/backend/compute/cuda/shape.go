//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "shape.h"
import "C"

import (
	"fmt"
	"math"
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

// Split returns equal-sized C-contiguous chunks along a logical dimension as one
// flat buffer. outer is the number of row-major blocks before the split
// dimension, dimSize is the full size of that dimension, splitSize is the size
// of each returned chunk along it, and inner is the number of elements after the
// split dimension. dimSize must be divisible by splitSize. The output ordering
// is outer blocks, then consecutive split chunks, then splitSize*inner elements
// within each chunk.
func (c *CUDAShapeOps) Split(
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
			"cuda_split: outer, dimSize, splitSize, and inner must be positive",
		)
	}

	if splitSize > dimSize {
		return nil, fmt.Errorf("cuda_split: splitSize %d exceeds dimSize %d", splitSize, dimSize)
	}

	if dimSize%splitSize != 0 {
		return nil, fmt.Errorf("cuda_split: dimSize %d is not divisible by splitSize %d", dimSize, splitSize)
	}

	maxInt := math.MaxInt

	if outer > maxInt/dimSize {
		return nil, fmt.Errorf("cuda_split: outer*dimSize overflows int")
	}

	expected := outer * dimSize

	if expected > maxInt/inner {
		return nil, fmt.Errorf("cuda_split: outer*dimSize*inner overflows int")
	}

	expected *= inner

	if expected != n {
		return nil, fmt.Errorf(
			"cuda_split: input length %d does not match outer*dimSize*inner=%d",
			n, expected,
		)
	}

	dst := make([]float64, n)
	rc := C.cuda_split(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(outer), C.int(dimSize), C.int(splitSize), C.int(inner),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_split failed (rc=%d)", rc)
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

func (c *CUDAShapeOps) LastToken(
	input []float64,
	outer int,
	sequenceLength int,
	featureLength int,
) ([]float64, error) {
	if outer <= 0 || sequenceLength <= 0 || featureLength <= 0 {
		return nil, fmt.Errorf("cuda_last_token: dimensions must be positive")
	}

	requiredInput := int64(outer) * int64(sequenceLength) * int64(featureLength)
	outputLength := int64(outer) * int64(featureLength)

	if requiredInput > int64(len(input)) {
		return nil, fmt.Errorf(
			"cuda_last_token: input length %d < required length %d",
			len(input),
			requiredInput,
		)
	}

	if outputLength > int64(math.MaxInt32) {
		return nil, fmt.Errorf("cuda_last_token: output length %d exceeds int32", outputLength)
	}

	dst := make([]float64, int(outputLength))
	rc := C.cuda_last_token(
		(*C.double)(unsafe.Pointer(&input[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(outer),
		C.int(sequenceLength),
		C.int(featureLength),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_last_token failed (rc=%d)", rc)
	}

	return dst, nil
}
