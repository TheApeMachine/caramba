//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "vsa.h"
import "C"

import (
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

/*
MetalVSAOps dispatches VSA operations via Metal.
Inputs are float64 on the Go side; kernels use float32.

Safe for concurrent use from multiple goroutines: a mutex serialises calls on this wrapper.

metallib must resolve to vsa.metallib (see Makefile in repo root).
*/
type MetalVSAOps struct {
	mu       sync.Mutex
	metallib string
}

/*
NewVSAOps initialises the Metal pipelines from the given .metallib path.
*/
func NewVSAOps(metallib string) (*MetalVSAOps, error) {
	if strings.TrimSpace(metallib) == "" {
		return nil, fmt.Errorf("NewVSAOps: metallib path is empty")
	}

	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_vsa_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_vsa_init failed (rc=%d): check %q exists", rc, metallib)
	}

	return &MetalVSAOps{metallib: metallib}, nil
}

/*
Close releases Metal objects held by the VSA bridge.
*/
func (metalVSAOps *MetalVSAOps) Close() error {
	metalVSAOps.mu.Lock()
	defer metalVSAOps.mu.Unlock()

	if rc := C.metal_vsa_cleanup(); rc != 0 {
		return fmt.Errorf("metal_vsa_cleanup failed (rc=%d)", rc)
	}

	return nil
}

/*
Bind computes elementwise product of data[0] and data[1].
shape=[N].
*/
func (metalVSAOps *MetalVSAOps) Bind(shape []int, data ...[]float64) ([]float64, error) {
	metalVSAOps.mu.Lock()
	defer metalVSAOps.mu.Unlock()

	if len(shape) < 1 || shape[0] <= 0 {
		return nil, fmt.Errorf("MetalVSAOps.Bind: need shape[0] > 0")
	}

	n := shape[0]

	if len(data) < 2 {
		return nil, fmt.Errorf("MetalVSAOps.Bind: need len(data) >= 2")
	}

	if len(data[0]) < n || len(data[1]) < n {
		return nil, fmt.Errorf("MetalVSAOps.Bind: vector lengths must be >= %d", n)
	}

	a := toFloat32(data[0])
	b := toFloat32(data[1])
	out := make([]float32, n)

	rc := C.metal_vsa_bind(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_vsa_bind failed (rc=%d)", rc)
	}

	return toFloat64(out), nil
}

/*
Bundle superimposes all input vectors (summed on host, then L2-normalised on GPU).
shape=[N].
*/
func (metalVSAOps *MetalVSAOps) Bundle(shape []int, data ...[]float64) ([]float64, error) {
	metalVSAOps.mu.Lock()
	defer metalVSAOps.mu.Unlock()

	if len(shape) < 1 || shape[0] <= 0 {
		return nil, fmt.Errorf("MetalVSAOps.Bundle: need shape[0] > 0")
	}

	n := shape[0]

	if len(data) < 1 {
		return nil, fmt.Errorf("MetalVSAOps.Bundle: need at least one input vector")
	}

	acc := make([]float32, n)

	for _, vec := range data {
		if len(vec) < n {
			return nil, fmt.Errorf("MetalVSAOps.Bundle: each vector must have length >= %d", n)
		}

		v32 := toFloat32(vec)

		for i := range acc {
			acc[i] += v32[i]
		}
	}

	out := make([]float32, n)

	rc := C.metal_vsa_l2normalize(
		(*C.float)(unsafe.Pointer(&acc[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_vsa_l2normalize failed (rc=%d)", rc)
	}

	return toFloat64(out), nil
}

/*
Similarity computes dot-product similarity between data[0] and data[1] on the GPU.
shape=[N], returns length-1 slice.
*/
func (metalVSAOps *MetalVSAOps) Similarity(shape []int, data ...[]float64) ([]float64, error) {
	metalVSAOps.mu.Lock()
	defer metalVSAOps.mu.Unlock()

	if len(shape) < 1 || shape[0] <= 0 {
		return nil, fmt.Errorf("MetalVSAOps.Similarity: need shape[0] > 0")
	}

	n := shape[0]

	if len(data) < 2 {
		return nil, fmt.Errorf("MetalVSAOps.Similarity: need len(data) >= 2")
	}

	if len(data[0]) < n || len(data[1]) < n {
		return nil, fmt.Errorf("MetalVSAOps.Similarity: vector lengths must be >= %d", n)
	}

	a := toFloat32(data[0])
	b := toFloat32(data[1])
	out := make([]float32, 1)

	rc := C.metal_vsa_dot(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_vsa_dot failed (rc=%d)", rc)
	}

	return toFloat64(out), nil
}
