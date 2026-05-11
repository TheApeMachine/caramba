//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "hawkes.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
MetalHawkes dispatches Hawkes process operations to the GPU via Metal.
*/
type MetalHawkes struct {
	metallib string
}

func NewHawkes(metallib string) (*MetalHawkes, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_hawkes_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_hawkes_init failed (rc=%d): check %q exists", rc, metallib)
	}

	return &MetalHawkes{metallib: metallib}, nil
}

// Intensity: shape=[K,T], data[0]=times[T], data[1]=alpha[K], data[2]=beta[K], data[3]=mu[K], data[4]=t[1].
func (op *MetalHawkes) Intensity(shape []int, data ...[]float64) ([]float64, error) {
	K, T := shape[0], shape[1]
	times := toFloat32(data[0])
	alpha := toFloat32(data[1])
	beta := toFloat32(data[2])
	mu := toFloat32(data[3])
	out := make([]float32, K)
	rc := C.metal_hawkes_intensity(
		(*C.float)(unsafe.Pointer(&times[0])),
		(*C.float)(unsafe.Pointer(&alpha[0])),
		(*C.float)(unsafe.Pointer(&beta[0])),
		(*C.float)(unsafe.Pointer(&mu[0])),
		C.float(data[4][0]),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(K), C.int(T),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_hawkes_intensity failed (rc=%d)", rc)
	}
	return toFloat64(out), nil
}

// KernelMatrix: shape=[T], data[0]=times[T], data[1]=alpha[1], data[2]=beta[1].
func (op *MetalHawkes) KernelMatrix(shape []int, data ...[]float64) ([]float64, error) {
	T := shape[0]
	times := toFloat32(data[0])
	out := make([]float32, T*T)
	rc := C.metal_hawkes_kernel_matrix(
		(*C.float)(unsafe.Pointer(&times[0])),
		C.float(data[1][0]),
		C.float(data[2][0]),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(T),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_hawkes_kernel_matrix failed (rc=%d)", rc)
	}
	return toFloat64(out), nil
}

// LogLikelihood: shape=[T], data[0]=times[T], data[1]=intensities[T], data[2]=integral[1].
func (op *MetalHawkes) LogLikelihood(shape []int, data ...[]float64) ([]float64, error) {
	T := shape[0]
	intensities := toFloat32(data[1])
	out := make([]float32, 1)
	rc := C.metal_hawkes_log_likelihood(
		(*C.float)(unsafe.Pointer(&intensities[0])),
		C.float(data[2][0]),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(T),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_hawkes_log_likelihood failed (rc=%d)", rc)
	}
	return toFloat64(out), nil
}

// Simulate: shape=[K,maxSteps], data[0]=mu[K], data[1]=alpha[K], data[2]=beta[K], data[3]=T_max[1].
func (op *MetalHawkes) Simulate(shape []int, data ...[]float64) ([]float64, error) {
	K, maxSteps := shape[0], shape[1]
	mu := toFloat32(data[0])
	alpha := toFloat32(data[1])
	beta := toFloat32(data[2])
	out := make([]float32, K*maxSteps)
	rc := C.metal_hawkes_simulate(
		(*C.float)(unsafe.Pointer(&mu[0])),
		(*C.float)(unsafe.Pointer(&alpha[0])),
		(*C.float)(unsafe.Pointer(&beta[0])),
		C.float(data[3][0]),
		C.int(K), C.int(maxSteps),
		(*C.float)(unsafe.Pointer(&out[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("metal_hawkes_simulate failed (rc=%d)", rc)
	}
	return toFloat64(out), nil
}
