//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_vsa.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
XLAVSAOps dispatches VSA (Vector Symbolic Algebra) operations to the XLA runtime via PJRT.
Operations are expressed as StableHLO modules, compiled once and cached for reuse.
*/
type XLAVSAOps struct {
	platform string
}

/*
NewVSAOps initialises the PJRT client for the given platform ("cpu"/"gpu").
*/
func NewVSAOps(platform string) (*XLAVSAOps, error) {
	config, err := NewPJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	if err := config.ValidateRuntime(); err != nil {
		return nil, err
	}

	cp := C.CString(platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_vsa_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_vsa_init failed for platform %q", platform)
	}

	return &XLAVSAOps{platform: platform}, nil
}

/*
Shutdown releases all PJRT VSA resources.
*/
func (xlaVSAOps *XLAVSAOps) Shutdown() { C.xla_vsa_shutdown() }

/*
Bind computes elementwise product of data[0] and data[1] via XLA.
shape=[N].
*/
func (xlaVSAOps *XLAVSAOps) Bind(shape []int, data ...[]float64) []float64 {
	n := shape[0]
	out := make([]float64, n)

	rc := C.xla_vsa_bind(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		panic("xla_vsa_bind failed")
	}

	return out
}

/*
Bundle superimposes all input vectors and returns an L2-normalised result via XLA.
shape=[N].
*/
func (xlaVSAOps *XLAVSAOps) Bundle(shape []int, data ...[]float64) []float64 {
	n := shape[0]
	numVecs := len(data)

	ptrs := make([]*C.double, numVecs)

	for i, vec := range data {
		ptrs[i] = (*C.double)(unsafe.Pointer(&vec[0]))
	}

	out := make([]float64, n)

	rc := C.xla_vsa_bundle(
		(**C.double)(unsafe.Pointer(&ptrs[0])),
		C.int(numVecs),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		panic("xla_vsa_bundle failed")
	}

	return out
}

/*
Similarity computes dot-product similarity between data[0] and data[1] via XLA.
shape=[N], returns length-1 slice.
*/
func (xlaVSAOps *XLAVSAOps) Similarity(shape []int, data ...[]float64) []float64 {
	n := shape[0]
	out := make([]float64, 1)

	rc := C.xla_vsa_similarity(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		panic("xla_vsa_similarity failed")
	}

	return out
}
