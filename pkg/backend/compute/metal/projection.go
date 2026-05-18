//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "projection.h"
import "C"

import (
	"fmt"
	"math"
	"math/rand"
	"unsafe"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

// ProjectionOps dispatches projection kernels (Linear, FusedQKV, TiedEmbedding)
// to the GPU via Metal.
type ProjectionOps struct {
	metallib string
	runtime  *MetalRuntime
}

// NewProjectionOps initialises Metal and loads projection.metallib.
func NewProjectionOps(metallib string) (*ProjectionOps, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_projection_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_projection_init failed (rc=%d): check that %q exists", rc, metallib)
	}
	runtime, err := newStandaloneMetalRuntime()
	if err != nil {
		return nil, err
	}

	return &ProjectionOps{metallib: metallib, runtime: runtime}, nil
}

// ---------------------------------------------------------------------------
// Kaiming-uniform weight initialisation helpers (same as CPU layer)
// ---------------------------------------------------------------------------

// InitLinearWeights returns Kaiming-uniform weights [outFeatures*inFeatures]
// and bias [outFeatures] initialised uniformly in [-1/sqrt(inFeatures), 1/sqrt(inFeatures)].
func InitLinearWeights(inFeatures, outFeatures int) ([]float64, []float64) {
	bound := math.Sqrt(2.0 / float64(inFeatures))
	w := make([]float64, outFeatures*inFeatures)
	for i := range w {
		w[i] = (rand.Float64()*2 - 1) * bound
	}
	bBound := 1.0 / math.Sqrt(float64(inFeatures))
	b := make([]float64, outFeatures)
	for i := range b {
		b[i] = (rand.Float64()*2 - 1) * bBound
	}
	return w, b
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

// Linear computes output = x @ weight + bias.
// weight [inFeatures*outFeatures] row-major.
// bias   [outFeatures] or nil.
// shape  = [M, inFeatures].
// data[0] = x.
func (p *ProjectionOps) Linear(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	M := shape[0]
	K := shape[1]
	N := len(weight) / K
	if N == 0 {
		return nil, fmt.Errorf("metal Linear: weight length %d not divisible by K=%d", len(weight), K)
	}

	src32 := toFloat32(data[0])
	w32 := toFloat32(weight)
	dst32 := make([]float32, M*N)

	var b32Ptr *C.float
	var b32 []float32
	if bias != nil {
		b32 = toFloat32(bias)
		b32Ptr = (*C.float)(unsafe.Pointer(&b32[0]))
	}

	rc := C.metal_linear(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&w32[0])),
		b32Ptr,
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(M), C.int(K), C.int(N),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_linear failed (rc=%d)", rc)
	}
	return toFloat64(dst32), nil
}

// Forward dispatches Linear with default Kaiming weights (for interface compliance).
// For production use, call Linear directly.
func (p *ProjectionOps) Forward(shape []int, data ...[]float64) ([]float64, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf(
			"metal projection Forward: insufficient input vectors (need data[0]=x, data[1]=weight; got %d slices)",
			len(data),
		)
	}

	var bias []float64

	if len(data) >= 3 {
		bias = data[2]
	}

	return p.Linear(shape, data[1], bias, data[0])
}

// ---------------------------------------------------------------------------
// FusedQKV
// ---------------------------------------------------------------------------

// FusedQKV computes x @ weight [+bias] where weight is the combined QKV weight.
// weight [dIn*outDim] where outDim = DQ+DK+DV.
// Returns [M * outDim].
func (p *ProjectionOps) FusedQKV(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	M := shape[0]
	K := shape[1]
	N := len(weight) / K
	if N == 0 {
		return nil, fmt.Errorf("metal FusedQKV: weight length %d not divisible by K=%d", len(weight), K)
	}

	src32 := toFloat32(data[0])
	w32 := toFloat32(weight)
	dst32 := make([]float32, M*N)

	var b32Ptr *C.float
	var b32 []float32
	if bias != nil {
		b32 = toFloat32(bias)
		b32Ptr = (*C.float)(unsafe.Pointer(&b32[0]))
	}

	rc := C.metal_fused_qkv(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&w32[0])),
		b32Ptr,
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(M), C.int(K), C.int(N),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_fused_qkv failed (rc=%d)", rc)
	}
	return toFloat64(dst32), nil
}

/*
FusedQKVTensor computes resident fused QKV projection.
*/
func (projectionOps *ProjectionOps) FusedQKVTensor(
	input, weight, bias computetensor.Tensor,
	outputShape computetensor.Shape,
	rows, inFeatures, outFeatures int,
) (computetensor.Tensor, error) {
	metalInput, err := requireMetalTensor(input)
	if err != nil {
		return nil, err
	}

	metalWeight, err := requireMetalTensor(weight)
	if err != nil {
		return nil, err
	}

	var metalBias *Tensor
	if bias != nil {
		metalBias, err = requireMetalTensor(bias)
		if err != nil {
			return nil, err
		}
	}

	if rows <= 0 || inFeatures <= 0 || outFeatures <= 0 {
		return nil, fmt.Errorf("metal tensor: fused_qkv dimensions must be positive")
	}

	if metalInput.Len() != rows*inFeatures ||
		metalWeight.Len() != inFeatures*outFeatures ||
		outputShape.Len() != rows*outFeatures {
		return nil, fmt.Errorf("metal tensor: fused_qkv shape mismatch")
	}

	var biasBuffer unsafe.Pointer
	if metalBias != nil {
		if metalBias.Len() != outFeatures {
			return nil, fmt.Errorf("metal tensor: fused_qkv bias length mismatch")
		}

		biasBuffer = metalBias.buffer
	}

	output, err := projectionOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_fused_qkv_tensor(
		metalInput.buffer,
		metalWeight.buffer,
		biasBuffer,
		output.buffer,
		C.int(rows),
		C.int(inFeatures),
		C.int(outFeatures),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: fused_qkv launch failed")
	}

	return output, nil
}

// ---------------------------------------------------------------------------
// TiedEmbedding
// ---------------------------------------------------------------------------

// TiedEmbedding computes logits = hidden @ weight.
// weight [dModel*vocabSize].
// shape  = [batch, seq, dModel] or [M, dModel].
// Returns [M * vocabSize].
func (p *ProjectionOps) TiedEmbedding(shape []int, weight []float64, data ...[]float64) ([]float64, error) {
	M := 1
	for i := 0; i < len(shape)-1; i++ {
		M *= shape[i]
	}
	D := shape[len(shape)-1]
	V := len(weight) / D
	if V == 0 {
		return nil, fmt.Errorf("metal TiedEmbedding: weight length %d not divisible by D=%d", len(weight), D)
	}

	src32 := toFloat32(data[0])
	w32 := toFloat32(weight)
	dst32 := make([]float32, M*V)

	rc := C.metal_tied_embedding(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&w32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(M), C.int(D), C.int(V),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_tied_embedding failed (rc=%d)", rc)
	}
	return toFloat64(dst32), nil
}
