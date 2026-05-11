//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_hawkes.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
XLAHawkes dispatches Hawkes process operations to XLA via PJRT.
*/
type XLAHawkes struct {
	platform string
}

func NewHawkes(platform string) (*XLAHawkes, error) {
	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_hawkes_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_hawkes_init failed for platform %q", platform)
	}

	return &XLAHawkes{platform: platform}, nil
}

func (op *XLAHawkes) Shutdown() { C.xla_hawkes_shutdown() }

// Intensity: shape=[K,T], data[0]=times[T], data[1]=alpha[K], data[2]=beta[K], data[3]=mu[K], data[4]=t[1].
func (op *XLAHawkes) Intensity(shape []int, data ...[]float64) []float64 {
	K, T := shape[0], shape[1]
	out := make([]float64, K)
	rc := C.xla_hawkes_intensity(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&data[3][0])),
		C.double(data[4][0]),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(K), C.int(T),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_hawkes_intensity failed"))
	}
	return out
}

// KernelMatrix: shape=[T], data[0]=times[T], data[1]=alpha[1], data[2]=beta[1].
func (op *XLAHawkes) KernelMatrix(shape []int, data ...[]float64) []float64 {
	T := shape[0]
	out := make([]float64, T*T)
	rc := C.xla_hawkes_kernel_matrix(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		C.double(data[1][0]),
		C.double(data[2][0]),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(T),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_hawkes_kernel_matrix failed"))
	}
	return out
}

// LogLikelihood: shape=[T], data[0]=times[T], data[1]=intensities[T], data[2]=integral[1].
func (op *XLAHawkes) LogLikelihood(shape []int, data ...[]float64) []float64 {
	T := shape[0]
	out := make([]float64, 1)
	rc := C.xla_hawkes_log_likelihood(
		(*C.double)(unsafe.Pointer(&data[1][0])),
		C.double(data[2][0]),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(T),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_hawkes_log_likelihood failed"))
	}
	return out
}

// Simulate: shape=[K,maxSteps], data[0]=mu[K], data[1]=alpha[K], data[2]=beta[K], data[3]=T_max[1].
func (op *XLAHawkes) Simulate(shape []int, data ...[]float64) []float64 {
	K, maxSteps := shape[0], shape[1]
	out := make([]float64, K*maxSteps)
	rc := C.xla_hawkes_simulate(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		C.double(data[3][0]),
		C.int(K), C.int(maxSteps),
		(*C.double)(unsafe.Pointer(&out[0])),
	)
	if rc != 0 {
		panic(fmt.Sprintf("xla_hawkes_simulate failed"))
	}
	return out
}
