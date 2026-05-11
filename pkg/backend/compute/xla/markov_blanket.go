//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_markov_blanket.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
XLAMarkovBlanket dispatches Markov blanket operations to XLA via PJRT.
*/
type XLAMarkovBlanket struct {
	platform string
}

func NewMarkovBlanket(platform string) (*XLAMarkovBlanket, error) {
	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_mb_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_mb_init failed for platform %q", platform)
	}

	return &XLAMarkovBlanket{platform: platform}, nil
}

func (op *XLAMarkovBlanket) Shutdown() { C.xla_mb_shutdown() }

// Partition: shape=[N,Ns,Na,Ni,Ne], data[0]=x, data[1]=masks[4*N].
func (op *XLAMarkovBlanket) Partition(shape []int, data ...[]float64) []float64 {
	N, Ns, Na, Ni, Ne := shape[0], shape[1], shape[2], shape[3], shape[4]
	out := make([]float64, Ns+Na+Ni+Ne)
	rc := C.xla_mb_partition(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(N), C.int(Ns), C.int(Na), C.int(Ni), C.int(Ne),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_mb_partition failed"))
	}
	return out
}

// FlowInternal: shape=[Ni,Ns], data[0]=x_sens, data[1]=W, data[2]=bias.
func (op *XLAMarkovBlanket) FlowInternal(shape []int, data ...[]float64) []float64 {
	Ni, Ns := shape[0], shape[1]
	out := make([]float64, Ni)
	rc := C.xla_mb_flow_internal(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(Ni), C.int(Ns),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_mb_flow_internal failed"))
	}
	return out
}

// FlowActive: shape=[Na,Ni], data[0]=x_int, data[1]=W, data[2]=bias.
func (op *XLAMarkovBlanket) FlowActive(shape []int, data ...[]float64) []float64 {
	Na, Ni := shape[0], shape[1]
	out := make([]float64, Na)
	rc := C.xla_mb_flow_active(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(Na), C.int(Ni),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_mb_flow_active failed"))
	}
	return out
}

// MutualInformation: shape=[N,M], data[0]=X[T*N], data[1]=Y[T*M].
func (op *XLAMarkovBlanket) MutualInformation(shape []int, data ...[]float64) []float64 {
	N, M := shape[0], shape[1]
	T := len(data[0]) / N
	out := make([]float64, 1)
	rc := C.xla_mb_mutual_information(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(T), C.int(N), C.int(M),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_mb_mutual_information failed"))
	}
	return out
}
