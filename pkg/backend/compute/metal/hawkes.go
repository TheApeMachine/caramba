//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "hawkes.h"
import "C"

import (
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

/*
MetalHawkes dispatches Hawkes process operations via Metal compute kernels (intensity,
kernel matrix, log-likelihood terms, and Ogata simulation).

All API methods convert float64 slices to float32 before the C layer; results are
float64 again. Values with magnitude beyond float32 precision may be rounded — use
CPU/CUDA paths if you require strict float64 end-to-end.

Safe for concurrent use from multiple goroutines: a mutex serialises this wrapper.

metallib must be hawkes.metallib from the repo Makefile.
*/
type MetalHawkes struct {
	mu       sync.Mutex
	metallib string
	runtime  *MetalRuntime
}

func NewHawkes(metallib string) (*MetalHawkes, error) {
	if strings.TrimSpace(metallib) == "" {
		return nil, fmt.Errorf("NewHawkes: metallib path is empty")
	}

	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_hawkes_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_hawkes_init failed (rc=%d): check %q exists", rc, metallib)
	}

	runtime, err := newStandaloneMetalRuntime()
	if err != nil {
		return nil, err
	}

	return &MetalHawkes{metallib: metallib, runtime: runtime}, nil
}

/*
Close releases Hawkes init state.
*/
func (op *MetalHawkes) Close() error {
	op.mu.Lock()
	defer op.mu.Unlock()

	if rc := C.metal_hawkes_cleanup(); rc != 0 {
		return fmt.Errorf("metal_hawkes_cleanup failed (rc=%d)", rc)
	}

	return nil
}

// Intensity: shape=[K,T], data[0]=times[T], data[1]=alpha[K], data[2]=beta[K], data[3]=mu[K], data[4]=t[1].
func (op *MetalHawkes) Intensity(shape []int, data ...[]float64) ([]float64, error) {
	op.mu.Lock()
	defer op.mu.Unlock()

	if len(shape) < 2 {
		return nil, fmt.Errorf("MetalHawkes.Intensity: need len(shape) >= 2")
	}

	K, T := shape[0], shape[1]

	if K <= 0 || T < 0 {
		return nil, fmt.Errorf("MetalHawkes.Intensity: need K>0 and T>=0")
	}

	if len(data) < 5 {
		return nil, fmt.Errorf("MetalHawkes.Intensity: need len(data) >= 5")
	}

	if len(data[0]) < T || len(data[1]) < K || len(data[2]) < K || len(data[3]) < K || len(data[4]) < 1 {
		return nil, fmt.Errorf("MetalHawkes.Intensity: slice lengths too short")
	}

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
	op.mu.Lock()
	defer op.mu.Unlock()

	if len(shape) < 1 || shape[0] <= 0 {
		return nil, fmt.Errorf("MetalHawkes.KernelMatrix: need T>0")
	}

	T := shape[0]

	if len(data) < 3 {
		return nil, fmt.Errorf("MetalHawkes.KernelMatrix: need len(data) >= 3")
	}

	if len(data[0]) < T || len(data[1]) < 1 || len(data[2]) < 1 {
		return nil, fmt.Errorf("MetalHawkes.KernelMatrix: slice lengths too short")
	}

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

// LogLikelihood: shape=[T], data[1]=intensities[T], data[2]=integral[1] (scalar packed as one-element slice).
func (op *MetalHawkes) LogLikelihood(shape []int, data ...[]float64) ([]float64, error) {
	op.mu.Lock()
	defer op.mu.Unlock()

	if len(shape) < 1 || shape[0] <= 0 {
		return nil, fmt.Errorf("MetalHawkes.LogLikelihood: need T>0")
	}

	T := shape[0]

	if len(data) < 3 {
		return nil, fmt.Errorf("MetalHawkes.LogLikelihood: need len(data) >= 3 (intensities, integral)")
	}

	if len(data[1]) < T || len(data[2]) < 1 {
		return nil, fmt.Errorf("MetalHawkes.LogLikelihood: intensities length %d, need %d; integral needs 1 element", len(data[1]), T)
	}

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
	op.mu.Lock()
	defer op.mu.Unlock()

	if len(shape) < 2 {
		return nil, fmt.Errorf("MetalHawkes.Simulate: need len(shape) >= 2")
	}

	K, maxSteps := shape[0], shape[1]

	if K <= 0 || maxSteps <= 0 {
		return nil, fmt.Errorf("MetalHawkes.Simulate: need K>0 and maxSteps>0")
	}

	if len(data) < 4 {
		return nil, fmt.Errorf("MetalHawkes.Simulate: need len(data) >= 4")
	}

	if len(data[0]) < K || len(data[1]) < K || len(data[2]) < K || len(data[3]) < 1 {
		return nil, fmt.Errorf("MetalHawkes.Simulate: slice lengths too short")
	}

	if data[3][0] <= 0 {
		return nil, fmt.Errorf("MetalHawkes.Simulate: T_max must be > 0")
	}

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
