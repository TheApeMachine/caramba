//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "markov_blanket.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
CUDAMarkovBlanket dispatches Markov blanket operations to the GPU via CUDA.
*/
type CUDAMarkovBlanket struct{}

func NewMarkovBlanket() *CUDAMarkovBlanket { return &CUDAMarkovBlanket{} }

// Partition extracts state partitions from joint vector x.
// shape=[N,Ns,Na,Ni,Ne], data[0]=x, data[1]=masks[4*N]
func (op *CUDAMarkovBlanket) Partition(shape []int, data ...[]float64) []float64 {
	N, Ns, Na, Ni, Ne := shape[0], shape[1], shape[2], shape[3], shape[4]
	out := make([]float64, Ns+Na+Ni+Ne)
	rc := C.cuda_mb_partition(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(N), C.int(Ns), C.int(Na), C.int(Ni), C.int(Ne),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_mb_partition failed (rc=%d)", rc))
	}
	return out
}

// FlowInternal: out[Ni] = W[Ni*Ns] @ x_sens[Ns] + bias[Ni]
func (op *CUDAMarkovBlanket) FlowInternal(shape []int, data ...[]float64) []float64 {
	Ni, Ns := shape[0], shape[1]
	out := make([]float64, Ni)
	rc := C.cuda_mb_flow_internal(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(Ni), C.int(Ns),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_mb_flow_internal failed (rc=%d)", rc))
	}
	return out
}

// FlowActive: out[Na] = W[Na*Ni] @ x_int[Ni] + bias[Na]
func (op *CUDAMarkovBlanket) FlowActive(shape []int, data ...[]float64) []float64 {
	Na, Ni := shape[0], shape[1]
	out := make([]float64, Na)
	rc := C.cuda_mb_flow_active(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(Na), C.int(Ni),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_mb_flow_active failed (rc=%d)", rc))
	}
	return out
}

// MutualInformation: shape=[N,M], data[0]=X[T*N], data[1]=Y[T*M]. Returns scalar.
func (op *CUDAMarkovBlanket) MutualInformation(shape []int, data ...[]float64) []float64 {
	N, M := shape[0], shape[1]
	T := len(data[0]) / N
	out := make([]float64, 1)
	rc := C.cuda_mb_mutual_information(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(T), C.int(N), C.int(M),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_mb_mutual_information failed (rc=%d)", rc))
	}
	return out
}
