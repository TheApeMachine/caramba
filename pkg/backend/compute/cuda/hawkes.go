//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "hawkes.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
CUDAHawkes dispatches Hawkes process operations to the GPU via CUDA.
*/
type CUDAHawkes struct{}

func NewHawkes() *CUDAHawkes { return &CUDAHawkes{} }

// Intensity: shape=[K,T], data[0]=times[T], data[1]=alpha[K], data[2]=beta[K], data[3]=mu[K], data[4]=t[1].
func (op *CUDAHawkes) Intensity(shape []int, data ...[]float64) []float64 {
	K, T := shape[0], shape[1]
	out := make([]float64, K)
	rc := C.cuda_hawkes_intensity(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&data[3][0])),
		C.double(data[4][0]),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(K), C.int(T),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_hawkes_intensity failed (rc=%d)", rc))
	}
	return out
}

// KernelMatrix: shape=[T], data[0]=times[T], data[1]=alpha[1], data[2]=beta[1]. Returns [T*T].
func (op *CUDAHawkes) KernelMatrix(shape []int, data ...[]float64) []float64 {
	T := shape[0]
	out := make([]float64, T*T)
	rc := C.cuda_hawkes_kernel_matrix(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		C.double(data[1][0]),
		C.double(data[2][0]),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(T),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_hawkes_kernel_matrix failed (rc=%d)", rc))
	}
	return out
}

// LogLikelihood: shape=[T], data[0]=times[T], data[1]=intensities[T], data[2]=integral[1].
func (op *CUDAHawkes) LogLikelihood(shape []int, data ...[]float64) []float64 {
	T := shape[0]
	out := make([]float64, 1)
	rc := C.cuda_hawkes_log_likelihood(
		(*C.double)(unsafe.Pointer(&data[1][0])),
		C.double(data[2][0]),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(T),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_hawkes_log_likelihood failed (rc=%d)", rc))
	}
	return out
}

// Simulate: shape=[K,maxSteps], data[0]=mu[K], data[1]=alpha[K], data[2]=beta[K], data[3]=T_max[1].
func (op *CUDAHawkes) Simulate(shape []int, data ...[]float64) []float64 {
	K, maxSteps := shape[0], shape[1]
	out := make([]float64, K*maxSteps)
	rc := C.cuda_hawkes_simulate(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		C.double(data[3][0]),
		C.int(K), C.int(maxSteps),
		(*C.double)(unsafe.Pointer(&out[0])),
	)
	if rc != 0 {
		panic(fmt.Sprintf("cuda_hawkes_simulate failed (rc=%d)", rc))
	}
	return out
}
