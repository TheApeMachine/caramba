//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_hawkes.h"
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

/*
XLAHawkes dispatches Hawkes process operations to XLA via PJRT.
*/
type XLAHawkes struct{}

func NewHawkes(platform string) (*XLAHawkes, error) {
	config, err := newRuntimePJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	cp := C.CString(config.Platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_hawkes_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_hawkes_init failed for platform %q (rc=%d)", config.Platform, rc)
	}

	return &XLAHawkes{}, nil
}

func (op *XLAHawkes) Shutdown() { C.xla_hawkes_shutdown() }

func cIntHawkes(name string, v int) (C.int, error) {
	if v < 0 || v > math.MaxInt32 {
		return 0, fmt.Errorf("%s: value %d out of range for C.int", name, v)
	}

	return C.int(int32(v)), nil
}

// Intensity: shape=[K,T], data[0]=times[T], data[1]=alpha[K], data[2]=beta[K], data[3]=mu[K], data[4]=t[1].
func (op *XLAHawkes) Intensity(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) < 2 {
		return nil, fmt.Errorf("XLAHawkes.Intensity: len(shape) < 2")
	}

	K, T := shape[0], shape[1]

	if K <= 0 || T < 0 || K > math.MaxInt32 || T > math.MaxInt32 {
		return nil, fmt.Errorf("XLAHawkes.Intensity: invalid K=%d T=%d", K, T)
	}

	if len(data) < 5 {
		return nil, fmt.Errorf("XLAHawkes.Intensity: len(data) < 5")
	}

	if len(data[0]) < T || len(data[1]) < K || len(data[2]) < K || len(data[3]) < K || len(data[4]) < 1 {
		return nil, fmt.Errorf("XLAHawkes.Intensity: data slice length mismatch")
	}

	cK, err := cIntHawkes("Intensity.K", K)
	if err != nil {
		return nil, err
	}

	cT, err := cIntHawkes("Intensity.T", T)
	if err != nil {
		return nil, err
	}

	out := make([]float64, K)
	rc := C.xla_hawkes_intensity(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		(*C.double)(unsafe.Pointer(&data[3][0])),
		C.double(data[4][0]),
		(*C.double)(unsafe.Pointer(&out[0])),
		cK, cT,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, fmt.Errorf("xla_hawkes_intensity failed (rc=%d)", rc)
	}

	return out, nil
}

// KernelMatrix: shape=[T], data[0]=times[T], data[1]=alpha[1], data[2]=beta[1].
func (op *XLAHawkes) KernelMatrix(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) != 1 {
		return nil, fmt.Errorf("XLAHawkes.KernelMatrix: len(shape) != 1")
	}

	T := shape[0]

	if T <= 0 || T > math.MaxInt32 {
		return nil, fmt.Errorf("XLAHawkes.KernelMatrix: invalid T=%d", T)
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("XLAHawkes.KernelMatrix: len(data) < 3")
	}

	if len(data[0]) < T || len(data[1]) < 1 || len(data[2]) < 1 {
		return nil, fmt.Errorf("XLAHawkes.KernelMatrix: data slice length mismatch")
	}

	if int64(T)*int64(T) > int64(math.MaxInt) {
		return nil, fmt.Errorf("XLAHawkes.KernelMatrix: T*T overflows")
	}

	out := make([]float64, T*T)
	cT, err := cIntHawkes("KernelMatrix.T", T)
	if err != nil {
		return nil, err
	}

	rc := C.xla_hawkes_kernel_matrix(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		C.double(data[1][0]),
		C.double(data[2][0]),
		(*C.double)(unsafe.Pointer(&out[0])),
		cT,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, fmt.Errorf("xla_hawkes_kernel_matrix failed (rc=%d)", rc)
	}

	return out, nil
}

// LogLikelihood: shape=[T], data[0]=times[T] (reserved / undocumented for this XLA entrypoint), data[1]=intensities[T], data[2]=integral[1].
// The C bridge xla_hawkes_log_likelihood uses only intensities and integral.
func (op *XLAHawkes) LogLikelihood(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) < 1 {
		return nil, fmt.Errorf("XLAHawkes.LogLikelihood: len(shape) < 1")
	}

	T := shape[0]

	if T < 0 || T > math.MaxInt32 {
		return nil, fmt.Errorf("XLAHawkes.LogLikelihood: invalid T=%d", T)
	}

	if len(data) < 3 {
		return nil, fmt.Errorf("XLAHawkes.LogLikelihood: len(data) < 3")
	}

	if len(data[0]) < T || len(data[1]) < T || len(data[2]) < 1 {
		return nil, fmt.Errorf("XLAHawkes.LogLikelihood: data slice length mismatch")
	}

	out := make([]float64, 1)
	cT, err := cIntHawkes("LogLikelihood.T", T)
	if err != nil {
		return nil, err
	}

	rc := C.xla_hawkes_log_likelihood(
		(*C.double)(unsafe.Pointer(&data[1][0])),
		C.double(data[2][0]),
		(*C.double)(unsafe.Pointer(&out[0])),
		cT,
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, fmt.Errorf("xla_hawkes_log_likelihood failed (rc=%d)", rc)
	}

	return out, nil
}

// Simulate: shape=[K,maxSteps], data[0]=mu[K], data[1]=alpha[K], data[2]=beta[K], data[3]=T_max[1].
func (op *XLAHawkes) Simulate(shape []int, data ...[]float64) ([]float64, error) {
	if len(shape) < 2 {
		return nil, fmt.Errorf("XLAHawkes.Simulate: len(shape) < 2")
	}

	K, maxSteps := shape[0], shape[1]

	if K <= 0 || maxSteps <= 0 || K > math.MaxInt32 || maxSteps > math.MaxInt32 {
		return nil, fmt.Errorf("XLAHawkes.Simulate: invalid K=%d maxSteps=%d", K, maxSteps)
	}

	if len(data) < 4 {
		return nil, fmt.Errorf("XLAHawkes.Simulate: len(data) < 4")
	}

	if len(data[0]) < K || len(data[1]) < K || len(data[2]) < K || len(data[3]) < 1 {
		return nil, fmt.Errorf("XLAHawkes.Simulate: data slice length mismatch")
	}

	if int64(K)*int64(maxSteps) > int64(math.MaxInt) {
		return nil, fmt.Errorf("XLAHawkes.Simulate: K*maxSteps overflows")
	}

	out := make([]float64, K*maxSteps)
	cK, err := cIntHawkes("Simulate.K", K)
	if err != nil {
		return nil, err
	}

	cSteps, err := cIntHawkes("Simulate.maxSteps", maxSteps)
	if err != nil {
		return nil, err
	}

	rc := C.xla_hawkes_simulate(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&data[2][0])),
		C.double(data[3][0]),
		cK, cSteps,
		(*C.double)(unsafe.Pointer(&out[0])),
	)
	runtime.KeepAlive(data)
	runtime.KeepAlive(out)

	if rc != 0 {
		return nil, fmt.Errorf("xla_hawkes_simulate failed (rc=%d)", rc)
	}

	return out, nil
}
