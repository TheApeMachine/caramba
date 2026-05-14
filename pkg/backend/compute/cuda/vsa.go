//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "vsa.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
CUDAVSAOps dispatches VSA (Vector Symbolic Algebra) operations to the GPU via CUDA.
All inputs are host pointers; device memory is managed internally by the kernels.
*/
type CUDAVSAOps struct{}

/*
NewVSAOps creates a CUDAVSAOps.
*/
func NewVSAOps() *CUDAVSAOps { return &CUDAVSAOps{} }

/*
Bind computes elementwise product of data[0] and data[1] on the GPU.
shape=[N].
*/
func (cudaVSAOps *CUDAVSAOps) Bind(shape []int, data ...[]float64) ([]float64, error) {
	n := shape[0]
	out := make([]float64, n)

	rc := C.cuda_vsa_bind(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_vsa_bind failed (rc=%d)", rc)
	}

	return out, nil
}

/*
Bundle superimposes all input vectors on the GPU and returns an L2-normalised result.
shape=[N].
*/
func (cudaVSAOps *CUDAVSAOps) Bundle(shape []int, data ...[]float64) ([]float64, error) {
	n := shape[0]
	numVecs := len(data)

	ptrs := make([]*C.double, numVecs)

	for i, vec := range data {
		ptrs[i] = (*C.double)(unsafe.Pointer(&vec[0]))
	}

	out := make([]float64, n)

	rc := C.cuda_vsa_bundle(
		(**C.double)(unsafe.Pointer(&ptrs[0])),
		C.int(numVecs),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_vsa_bundle failed (rc=%d)", rc)
	}

	return out, nil
}

/*
Similarity computes the dot-product similarity between data[0] and data[1] on the GPU.
shape=[N], returns length-1 slice.
*/
func (cudaVSAOps *CUDAVSAOps) Similarity(shape []int, data ...[]float64) ([]float64, error) {
	n := shape[0]
	out := make([]float64, 1)

	rc := C.cuda_vsa_similarity(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&data[1][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_vsa_similarity failed (rc=%d)", rc)
	}

	return out, nil
}

/*
Permute cyclically shifts a vector by shift positions on the GPU.
*/
func (cudaVSAOps *CUDAVSAOps) Permute(
	shape []int,
	shift int,
	data ...[]float64,
) ([]float64, error) {
	n := shape[0]
	out := make([]float64, n)

	rc := C.cuda_vsa_permute(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
		C.int(shift),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_vsa_permute failed (rc=%d)", rc)
	}

	return out, nil
}

/*
InversePermute cyclically shifts a vector by -shift positions on the GPU.
*/
func (cudaVSAOps *CUDAVSAOps) InversePermute(
	shape []int,
	shift int,
	data ...[]float64,
) ([]float64, error) {
	n := shape[0]
	out := make([]float64, n)

	rc := C.cuda_vsa_inverse_permute(
		(*C.double)(unsafe.Pointer(&data[0][0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(n),
		C.int(shift),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_vsa_inverse_permute failed (rc=%d)", rc)
	}

	return out, nil
}
