//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "markov_blanket.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
MetalMarkovBlanket dispatches Markov blanket operations to the GPU via Metal.
metallib must be the absolute path to markov_blanket.metallib.
*/
type MetalMarkovBlanket struct {
	metallib string
}

func NewMarkovBlanket(metallib string) (*MetalMarkovBlanket, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_mb_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_mb_init failed (rc=%d): check %q exists", rc, metallib)
	}

	return &MetalMarkovBlanket{metallib: metallib}, nil
}

// Partition extracts state partitions. shape=[N,Ns,Na,Ni,Ne], data[0]=x, data[1]=masks[4*N].
func (op *MetalMarkovBlanket) Partition(shape []int, data ...[]float64) ([]float64, error) {
	N, Ns, Na, Ni, Ne := shape[0], shape[1], shape[2], shape[3], shape[4]
	x := toFloat32(data[0])
	masks := toFloat32(data[1])
	out := make([]float32, Ns+Na+Ni+Ne)
	rc := C.metal_mb_partition(
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&masks[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(N), C.int(Ns), C.int(Na), C.int(Ni), C.int(Ne),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_mb_partition failed (rc=%d)", rc)
	}
	return toFloat64(out), nil
}

// FlowInternal: out[Ni] = W[Ni*Ns] @ x_sens[Ns] + bias[Ni]
func (op *MetalMarkovBlanket) FlowInternal(shape []int, data ...[]float64) ([]float64, error) {
	Ni, Ns := shape[0], shape[1]
	xSens := toFloat32(data[0])
	w := toFloat32(data[1])
	bias := toFloat32(data[2])
	out := make([]float32, Ni)
	rc := C.metal_mb_flow_internal(
		(*C.float)(unsafe.Pointer(&xSens[0])),
		(*C.float)(unsafe.Pointer(&w[0])),
		(*C.float)(unsafe.Pointer(&bias[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(Ni), C.int(Ns),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_mb_flow_internal failed (rc=%d)", rc)
	}
	return toFloat64(out), nil
}

// FlowActive: out[Na] = W[Na*Ni] @ x_int[Ni] + bias[Na]
func (op *MetalMarkovBlanket) FlowActive(shape []int, data ...[]float64) ([]float64, error) {
	Na, Ni := shape[0], shape[1]
	xInt := toFloat32(data[0])
	w := toFloat32(data[1])
	bias := toFloat32(data[2])
	out := make([]float32, Na)
	rc := C.metal_mb_flow_active(
		(*C.float)(unsafe.Pointer(&xInt[0])),
		(*C.float)(unsafe.Pointer(&w[0])),
		(*C.float)(unsafe.Pointer(&bias[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(Na), C.int(Ni),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_mb_flow_active failed (rc=%d)", rc)
	}
	return toFloat64(out), nil
}

// MutualInformation: shape=[N,M], data[0]=X[T*N], data[1]=Y[T*M]. Returns scalar.
func (op *MetalMarkovBlanket) MutualInformation(shape []int, data ...[]float64) ([]float64, error) {
	N, M := shape[0], shape[1]
	T := len(data[0]) / N
	xData := toFloat32(data[0])
	yData := toFloat32(data[1])
	out := make([]float32, 1)
	rc := C.metal_mb_mutual_information(
		(*C.float)(unsafe.Pointer(&xData[0])),
		(*C.float)(unsafe.Pointer(&yData[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(T), C.int(N), C.int(M),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_mb_mutual_information failed (rc=%d)", rc)
	}
	return toFloat64(out), nil
}
