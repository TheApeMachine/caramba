//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "shape.h"
// #include "tensor.h"
import "C"

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

// MetalShapeOps dispatches shape manipulation kernels to the GPU via Metal.
type MetalShapeOps struct {
	metallib string
	runtime  *MetalRuntime
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

	runtime, err := newStandaloneMetalRuntime()
	if err != nil {
		return nil, err
	}

	return &MetalShapeOps{metallib: metallib, runtime: runtime}, nil
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
	case "shape.upsample_nearest2d":
		if len(data) != 1 {
			return nil, fmt.Errorf("metal shape Forward upsample_nearest2d: requires one input buffer")
		}

		if len(shape) != 4 {
			return nil, fmt.Errorf("metal shape Forward upsample_nearest2d: shape must be [B,C,H,W]")
		}

		scaleH := metalIntConfig(node, "scale_factor", 0)
		scaleW := metalIntConfig(node, "scale_factor", 0)
		scaleH = metalIntConfig(node, "scale_h", scaleH)
		scaleW = metalIntConfig(node, "scale_w", scaleW)

		return m.UpsampleNearest2D(
			data[0],
			shape[0],
			shape[1],
			shape[2],
			shape[3],
			scaleH,
			scaleW,
		)
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
	case "shape.last_token":
		if len(data) != 1 {
			return nil, fmt.Errorf("metal shape Forward last_token: requires one input buffer")
		}

		outer, sequenceLength, featureLength, err := lastTokenShapeParts(shape)

		if err != nil {
			return nil, err
		}

		return m.LastToken(data[0], outer, sequenceLength, featureLength)
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

func (m *MetalShapeOps) UpsampleNearest2D(
	input []float64,
	batch int,
	channels int,
	height int,
	width int,
	scaleH int,
	scaleW int,
) ([]float64, error) {
	if batch <= 0 || channels <= 0 || height <= 0 || width <= 0 ||
		scaleH <= 0 || scaleW <= 0 {
		return nil, fmt.Errorf("metal_upsample_nearest2d: dimensions must be positive")
	}

	inputLength := int64(batch) * int64(channels) * int64(height) * int64(width)
	outputLength := inputLength * int64(scaleH) * int64(scaleW)

	if inputLength != int64(len(input)) {
		return nil, fmt.Errorf(
			"metal_upsample_nearest2d: input length %d does not match NCHW size %d",
			len(input),
			inputLength,
		)
	}

	if outputLength > math.MaxInt32 {
		return nil, fmt.Errorf("metal_upsample_nearest2d: output length %d exceeds int32", outputLength)
	}

	src32 := toFloat32(input)
	dst32 := make([]float32, int(outputLength))
	rc := C.metal_upsample_nearest2d(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(batch),
		C.int(channels),
		C.int(height),
		C.int(width),
		C.int(scaleH),
		C.int(scaleW),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_upsample_nearest2d failed (rc=%d)", rc)
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

func (m *MetalShapeOps) LastToken(
	input []float64,
	outer int,
	sequenceLength int,
	featureLength int,
) ([]float64, error) {
	if outer <= 0 || sequenceLength <= 0 || featureLength <= 0 {
		return nil, fmt.Errorf("metal_last_token: dimensions must be positive")
	}

	requiredInput := int64(outer) * int64(sequenceLength) * int64(featureLength)
	outputLength := int64(outer) * int64(featureLength)

	if requiredInput > int64(len(input)) {
		return nil, fmt.Errorf(
			"metal_last_token: input length %d < required length %d",
			len(input),
			requiredInput,
		)
	}

	if outputLength > math.MaxInt32 {
		return nil, fmt.Errorf("metal_last_token: output length %d exceeds int32", outputLength)
	}

	src32 := toFloat32(input[:int(requiredInput)])
	dst32 := make([]float32, int(outputLength))
	rc := C.metal_last_token(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(outer),
		C.int(sequenceLength),
		C.int(featureLength),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_last_token failed (rc=%d)", rc)
	}

	return toFloat64(dst32), nil
}

func lastTokenShapeParts(shape []int) (int, int, int, error) {
	if len(shape) < 2 {
		return 0, 0, 0, fmt.Errorf("metal shape last_token: expected rank >= 2")
	}

	sequenceLength := shape[len(shape)-2]
	featureLength := shape[len(shape)-1]

	if sequenceLength <= 0 || featureLength <= 0 {
		return 0, 0, 0, fmt.Errorf("metal shape last_token: trailing dimensions must be positive")
	}

	outer := 1

	for _, dimension := range shape[:len(shape)-2] {
		if dimension <= 0 {
			return 0, 0, 0, fmt.Errorf("metal shape last_token: outer dimensions must be positive")
		}

		if outer > math.MaxInt/dimension {
			return 0, 0, 0, fmt.Errorf("metal shape last_token: shape product overflows int")
		}

		outer *= dimension
	}

	return outer, sequenceLength, featureLength, nil
}

func (m *MetalShapeOps) ViewAsHeadsTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
	B, T, H, headDim int,
) (computetensor.Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	if metalInput.Len() != B*T*H*headDim || outputShape.Len() != metalInput.Len() {
		return nil, fmt.Errorf("metal shape: invalid view_as_heads tensor shape")
	}

	output, err := m.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)

	if err != nil {
		return nil, err
	}

	rc := C.metal_view_as_heads_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(B),
		C.int(T),
		C.int(H),
		C.int(headDim),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_view_as_heads_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (m *MetalShapeOps) CopyTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	if metalInput.Len() != outputShape.Len() {
		return nil, fmt.Errorf("metal shape: reshape changes element count")
	}

	output, err := m.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)

	if err != nil {
		return nil, err
	}

	if outputShape.Len() == 0 {
		return output, nil
	}

	rc := C.metal_copy_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(outputShape.Len()),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_copy_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (m *MetalShapeOps) ReshapeTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	if metalInput.Len() != outputShape.Len() {
		return nil, fmt.Errorf("metal shape: reshape changes element count")
	}

	retained := C.metal_tensor_retain(metalInput.buffer)

	if retained == nil && outputShape.Len() > 0 {
		return nil, fmt.Errorf("metal shape: reshape retain failed")
	}

	bytes, err := outputShape.Bytes(metalInput.DType())

	if err != nil {
		if retained != nil {
			_ = C.metal_tensor_free(retained)
		}

		return nil, err
	}

	metadata := metalInput.Metadata()
	metadata.Shape = outputShape
	metadata.Strides = contiguousStrides(outputShape.Dims())
	metadata.ByteSize = bytes
	metadata.AliasOf = "reshape"

	return &Tensor{
		bytes:    bytes,
		shape:    outputShape,
		buffer:   retained,
		runtime:  m.runtime,
		metadata: metadata,
	}, nil
}

func (m *MetalShapeOps) TransposeTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
	dim0 int,
	dim1 int,
) (computetensor.Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	inputShape := metalInput.Shape().Dims()

	if len(inputShape) == 0 || len(inputShape) > 8 {
		return nil, fmt.Errorf("metal shape: transpose requires rank 1..8")
	}

	if dim0 < 0 || dim1 < 0 || dim0 >= len(inputShape) || dim1 >= len(inputShape) {
		return nil, fmt.Errorf("metal shape: transpose dimensions out of range")
	}

	if outputShape.Len() != metalInput.Len() {
		return nil, fmt.Errorf("metal shape: transpose changes element count")
	}

	outputDims := outputShape.Dims()
	expectedDims := append([]int(nil), inputShape...)
	expectedDims[dim0], expectedDims[dim1] = expectedDims[dim1], expectedDims[dim0]

	if !slices.Equal(outputDims, expectedDims) {
		return nil, fmt.Errorf(
			"metal shape: transpose output shape %v does not match expected %v",
			outputDims,
			expectedDims,
		)
	}

	shapeData := make([]C.int, len(inputShape))

	for dimensionIndex, dimension := range inputShape {
		shapeData[dimensionIndex] = C.int(dimension)
	}

	output, err := m.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)

	if err != nil {
		return nil, err
	}

	if outputShape.Len() == 0 {
		return output, nil
	}

	rc := C.metal_transpose_tensor(
		metalInput.buffer,
		output.buffer,
		(*C.int)(unsafe.Pointer(&shapeData[0])),
		C.int(len(shapeData)),
		C.int(dim0),
		C.int(dim1),
		C.int(outputShape.Len()),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_transpose_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
SlicePrefixTensor copies the first outputShape.Len() elements of
input into a fresh tensor sized outputShape. It is the Metal-side
implementation of shape.slice for the start=0 case — the FLUX-2
joint-stream → latent-only transition needs exactly this: take the
first N latent tokens after the joint dual blocks have processed
the concatenated latent+text sequence. A strided variant supporting
start>0 or interior dim slicing is the next thing to add when a
manifest needs it; until then operation_executor.applySlice rejects
those shapes with a clear error rather than silently miscopying.
*/
func (m *MetalShapeOps) SlicePrefixTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	if outputShape.Len() > metalInput.Len() {
		return nil, fmt.Errorf(
			"metal shape: slice output length %d exceeds input length %d",
			outputShape.Len(), metalInput.Len(),
		)
	}

	output, err := m.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)

	if err != nil {
		return nil, err
	}

	if outputShape.Len() == 0 {
		return output, nil
	}

	rc := C.metal_copy_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(outputShape.Len()),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_copy_tensor (slice) failed (rc=%d)", rc)
	}

	return output, nil
}

func (m *MetalShapeOps) ConcatTensor(
	left computetensor.Tensor,
	right computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	metalLeft, err := requireMetalTensor(left)

	if err != nil {
		return nil, err
	}

	metalRight, err := requireMetalTensor(right)

	if err != nil {
		return nil, err
	}

	if outputShape.Len() != metalLeft.Len()+metalRight.Len() {
		return nil, fmt.Errorf("metal shape: concat output shape has invalid length")
	}

	if metalLeft.Len() == 0 {
		return m.CopyTensor(right, outputShape)
	}

	if metalRight.Len() == 0 {
		return m.CopyTensor(left, outputShape)
	}

	output, err := m.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)

	if err != nil {
		return nil, err
	}

	rc := C.metal_concat_tensor(
		metalLeft.buffer,
		C.int(metalLeft.Len()),
		metalRight.buffer,
		C.int(metalRight.Len()),
		output.buffer,
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_concat_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (m *MetalShapeOps) SplitTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
	outer int,
	dimSize int,
	splitSize int,
	inner int,
) (computetensor.Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	if err := validateMetalSplitTensor(
		metalInput.Len(), outputShape.Len(), outer, dimSize, splitSize, inner,
	); err != nil {
		return nil, err
	}

	output, err := m.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)

	if err != nil {
		return nil, err
	}

	rc := C.metal_split_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(outer),
		C.int(dimSize),
		C.int(splitSize),
		C.int(inner),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_split_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (m *MetalShapeOps) MergeHeadsTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
	B, H, T, headDim int,
) (computetensor.Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	if metalInput.Len() != B*H*T*headDim || outputShape.Len() != metalInput.Len() {
		return nil, fmt.Errorf("metal shape: invalid merge_heads tensor shape")
	}

	output, err := m.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)

	if err != nil {
		return nil, err
	}

	rc := C.metal_merge_heads_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(B),
		C.int(H),
		C.int(T),
		C.int(headDim),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_merge_heads_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (m *MetalShapeOps) LastTokenTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
	outer, sequenceLength, featureLength int,
) (computetensor.Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	if metalInput.Len() < outer*sequenceLength*featureLength ||
		outputShape.Len() != outer*featureLength {
		return nil, fmt.Errorf("metal shape: invalid last_token tensor shape")
	}

	output, err := m.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)

	if err != nil {
		return nil, err
	}

	rc := C.metal_last_token_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(outer),
		C.int(sequenceLength),
		C.int(featureLength),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_last_token_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (m *MetalShapeOps) UpsampleNearest2DTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
	batch int,
	channels int,
	height int,
	width int,
	scaleH int,
	scaleW int,
) (computetensor.Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	inputLength := batch * channels * height * width
	outputLength := inputLength * scaleH * scaleW

	if metalInput.Len() != inputLength || outputShape.Len() != outputLength {
		return nil, fmt.Errorf("metal shape: invalid upsample_nearest2d tensor shape")
	}

	output, err := m.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)

	if err != nil {
		return nil, err
	}

	rc := C.metal_upsample_nearest2d_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(batch),
		C.int(channels),
		C.int(height),
		C.int(width),
		C.int(scaleH),
		C.int(scaleW),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_upsample_nearest2d_tensor failed (rc=%d)", rc)
	}

	return output, nil
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

func validateMetalSplitTensor(
	inputLength int,
	outputLength int,
	outer int,
	dimSize int,
	splitSize int,
	inner int,
) error {
	if outer <= 0 || dimSize <= 0 || splitSize <= 0 || inner <= 0 {
		return fmt.Errorf("metal shape: split dimensions must be positive")
	}

	if splitSize > dimSize || dimSize%splitSize != 0 {
		return fmt.Errorf("metal shape: invalid split size")
	}

	expected := outer * dimSize * inner

	if inputLength != expected || outputLength != expected {
		return fmt.Errorf("metal shape: invalid split tensor shape")
	}

	return nil
}
