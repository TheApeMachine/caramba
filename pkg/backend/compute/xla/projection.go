//go:build cgo && xla

package xla

// XLA projection backend via the PJRT C API.
//
// Build requirements (same as activation.go):
//   CGO_CPPFLAGS="-I/path/to/xla" CGO_LDFLAGS="-ldl -lstdc++"
//   go build -tags "cgo xla" ./backend/compute/xla/

// #cgo CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ldl -lstdc++
// #include "projection.h"
import "C"

import (
	"fmt"
	"math"
	"math/rand"
	"unsafe"
)

// XLAProjection dispatches projection operations to the XLA runtime via PJRT.
type XLAProjection struct {
	platform string
}

// NewXLAProjection initialises the PJRT projection client.
// Call xla.New(platform) for the activation client first; this reuses globals.
func NewXLAProjection(platform string) (*XLAProjection, error) {
	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_projection_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_projection_init failed for platform %q", platform)
	}
	return &XLAProjection{platform: platform}, nil
}

// Shutdown releases projection PJRT resources.
func (x *XLAProjection) Shutdown() { C.xla_projection_shutdown() }

// ---------------------------------------------------------------------------
// Weight initialisation helpers
// ---------------------------------------------------------------------------

// InitLinearWeights returns Kaiming-uniform weights and bias.
func InitXLALinearWeights(inFeatures, outFeatures int) ([]float64, []float64) {
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
func (x *XLAProjection) Linear(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	M := shape[0]
	K := shape[1]
	N := len(weight) / K
	dst := make([]float64, M*N)

	hasBias := 0
	var biasPtr *C.double
	if bias != nil {
		hasBias = 1
		biasPtr = (*C.double)(unsafe.Pointer(&bias[0]))
	}

	rc := C.xla_linear(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&weight[0])),
		biasPtr,
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(M), C.int(K), C.int(N), C.int(hasBias),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_linear failed")
	}
	return dst, nil
}

// Forward dispatches Linear. data[0]=x, data[1]=weight, data[2]=bias (optional).
func (x *XLAProjection) Forward(shape []int, data ...[]float64) []float64 {
	var bias []float64
	if len(data) >= 3 {
		bias = data[2]
	}
	out, _ := x.Linear(shape, data[1], bias, data[0])
	return out
}

// ---------------------------------------------------------------------------
// FusedQKV
// ---------------------------------------------------------------------------

// FusedQKV computes x @ weight^T [+bias].
func (x *XLAProjection) FusedQKV(shape []int, weight, bias []float64, data ...[]float64) ([]float64, error) {
	M := shape[0]
	K := shape[1]
	N := len(weight) / K
	dst := make([]float64, M*N)

	hasBias := 0
	var biasPtr *C.double
	if bias != nil {
		hasBias = 1
		biasPtr = (*C.double)(unsafe.Pointer(&bias[0]))
	}

	rc := C.xla_fused_qkv(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&weight[0])),
		biasPtr,
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(M), C.int(K), C.int(N), C.int(hasBias),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_fused_qkv failed")
	}
	return dst, nil
}

// ---------------------------------------------------------------------------
// TiedEmbedding
// ---------------------------------------------------------------------------

// TiedEmbedding computes logits = hidden @ weight^T.
// weight [vocabSize*dModel].
func (x *XLAProjection) TiedEmbedding(shape []int, weight []float64, data ...[]float64) ([]float64, error) {
	M := 1
	for i := 0; i < len(shape)-1; i++ {
		M *= shape[i]
	}
	D := shape[len(shape)-1]
	V := len(weight) / D
	dst := make([]float64, M*V)

	rc := C.xla_tied_embedding(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&weight[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(M), C.int(D), C.int(V),
	)
	if rc != 0 {
		return nil, fmt.Errorf("xla_tied_embedding failed")
	}
	return dst, nil
}
