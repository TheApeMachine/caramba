//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "projection.h"
import "C"

import (
	"fmt"
	"math"
	"math/rand"
	"unsafe"
)

// CUDAProjection dispatches projection kernels to the GPU via CUDA.
type CUDAProjection struct{}

// NewCUDAProjection creates a CUDAProjection.
func NewCUDAProjection() *CUDAProjection { return &CUDAProjection{} }

// ---------------------------------------------------------------------------
// Weight initialisation helpers
// ---------------------------------------------------------------------------

// InitLinearWeights returns Kaiming-uniform weights and bias.
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

// Linear computes output = x @ weight^T + bias.
// weight [outFeatures*inFeatures], bias [outFeatures] or nil.
// shape = [M, inFeatures].
func (c *CUDAProjection) Linear(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	M := shape[0]
	K := shape[1]
	N := len(weight) / K
	dst := make([]float64, M*N)

	var biasPtr *C.double
	if bias != nil {
		biasPtr = (*C.double)(unsafe.Pointer(&bias[0]))
	}
	hasBias := 0
	if bias != nil {
		hasBias = 1
	}

	rc := C.cuda_linear(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&weight[0])),
		biasPtr,
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(M), C.int(K), C.int(N), C.int(hasBias),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_linear failed (rc=%d)", rc)
	}
	return dst, nil
}

// Forward dispatches Linear. data[0]=x, data[1]=weight, data[2]=bias (optional).
func (c *CUDAProjection) Forward(shape []int, data ...[]float64) []float64 {
	var bias []float64
	if len(data) >= 3 {
		bias = data[2]
	}
	out, err := c.Linear(shape, data[1], bias, data[0])
	if err != nil {
		panic(err)
	}

	return out
}

// ---------------------------------------------------------------------------
// FusedQKV
// ---------------------------------------------------------------------------

// FusedQKV computes x @ weight^T [+bias], weight [(DQ+DK+DV)*DIn].
func (c *CUDAProjection) FusedQKV(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	M := shape[0]
	K := shape[1]
	N := len(weight) / K
	dst := make([]float64, M*N)

	var biasPtr *C.double
	if bias != nil {
		biasPtr = (*C.double)(unsafe.Pointer(&bias[0]))
	}
	hasBias := 0
	if bias != nil {
		hasBias = 1
	}

	rc := C.cuda_fused_qkv(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&weight[0])),
		biasPtr,
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(M), C.int(K), C.int(N), C.int(hasBias),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_fused_qkv failed (rc=%d)", rc)
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// TiedEmbedding
// ---------------------------------------------------------------------------

// TiedEmbedding computes logits = hidden @ weight^T.
// weight [vocabSize*dModel].
func (c *CUDAProjection) TiedEmbedding(shape []int, weight []float64, data ...[]float64) ([]float64, error) {
	M := 1
	for i := 0; i < len(shape)-1; i++ {
		M *= shape[i]
	}
	D := shape[len(shape)-1]
	V := len(weight) / D
	dst := make([]float64, M*V)

	rc := C.cuda_tied_embedding(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&weight[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(M), C.int(D), C.int(V),
	)
	if rc != 0 {
		return nil, fmt.Errorf("cuda_tied_embedding failed (rc=%d)", rc)
	}
	return dst, nil
}
