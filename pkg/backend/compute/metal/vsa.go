//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "vsa.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
MetalVSAOps dispatches VSA (Vector Symbolic Algebra) operations to the Apple GPU via Metal.
Inputs are float64 on the Go side; the Metal kernels operate in float32.
*/
type MetalVSAOps struct {
	metallib string
}

/*
NewVSAOps initialises the Metal pipelines from the given .metallib path.
*/
func NewVSAOps(metallib string) (*MetalVSAOps, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_vsa_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_vsa_init failed (rc=%d): check %q exists", rc, metallib)
	}

	return &MetalVSAOps{metallib: metallib}, nil
}

/*
Bind computes elementwise product of data[0] and data[1] on the GPU.
shape=[N].
*/
func (metalVSAOps *MetalVSAOps) Bind(shape []int, data ...[]float64) ([]float64, error) {
	n := shape[0]
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
	n := shape[0]
	acc := make([]float32, n)

	for _, vec := range data {
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
	n := shape[0]
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
