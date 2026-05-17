//go:build linux && cgo && cuda

package cuda

// #cgo LDFLAGS: -lcuda -lcudart
// #include "physics.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
CUDAPhysicsOps dispatches physics-stencil kernels (1D / 2D / 3D periodic
Laplacian) to the GPU via CUDA. Device memory is managed inside the C
wrappers; the Go-facing methods accept and return host slices to match
the other CUDA op types in this package.
*/
type CUDAPhysicsOps struct{}

/*
NewPhysics constructs the CUDA physics dispatcher. CUDA context creation
is lazy and happens on the first kernel launch.
*/
func NewPhysics() *CUDAPhysicsOps { return &CUDAPhysicsOps{} }

/*
Laplacian1D computes the 2nd-order central-difference Laplacian on a
length-n periodic grid: dst[i] = (src[(i-1+n)%n] + src[(i+1)%n] - 2*src[i]) * invH2.
*/
func (cudaPhysics *CUDAPhysicsOps) Laplacian1D(src []float64, n int, invH2 float64) ([]float64, error) {
	if n <= 0 {
		return nil, fmt.Errorf("cuda_laplacian_1d: non-positive n=%d", n)
	}

	if len(src) != n {
		return nil, fmt.Errorf("cuda_laplacian_1d: src length %d != n %d", len(src), n)
	}

	dst := make([]float64, n)

	rc := C.cuda_laplacian_1d(
		(*C.double)(unsafe.Pointer(&src[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
		C.double(invH2),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_laplacian_1d failed (rc=%d)", rc)
	}

	return dst, nil
}

/*
Laplacian2D computes the periodic 5-point Laplacian on a row-major
[H, W] grid.
*/
func (cudaPhysics *CUDAPhysicsOps) Laplacian2D(src []float64, h, w int, invH2 float64) ([]float64, error) {
	if h <= 0 || w <= 0 {
		return nil, fmt.Errorf("cuda_laplacian_2d: non-positive shape (%d,%d)", h, w)
	}

	total := h * w

	if len(src) != total {
		return nil, fmt.Errorf("cuda_laplacian_2d: src length %d != H*W %d", len(src), total)
	}

	dst := make([]float64, total)

	rc := C.cuda_laplacian_2d(
		(*C.double)(unsafe.Pointer(&src[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(h),
		C.int(w),
		C.double(invH2),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_laplacian_2d failed (rc=%d)", rc)
	}

	return dst, nil
}

/*
Laplacian3D computes the periodic 7-point Laplacian on a row-major
[D, H, W] grid.
*/
func (cudaPhysics *CUDAPhysicsOps) Laplacian3D(src []float64, d, h, w int, invH2 float64) ([]float64, error) {
	if d <= 0 || h <= 0 || w <= 0 {
		return nil, fmt.Errorf("cuda_laplacian_3d: non-positive shape (%d,%d,%d)", d, h, w)
	}

	total := d * h * w

	if len(src) != total {
		return nil, fmt.Errorf("cuda_laplacian_3d: src length %d != D*H*W %d", len(src), total)
	}

	dst := make([]float64, total)

	rc := C.cuda_laplacian_3d(
		(*C.double)(unsafe.Pointer(&src[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(d),
		C.int(h),
		C.int(w),
		C.double(invH2),
	)

	if rc != 0 {
		return nil, fmt.Errorf("cuda_laplacian_3d failed (rc=%d)", rc)
	}

	return dst, nil
}

/*
Laplacian dispatches to the rank-specific kernel based on the input
shape. Used by the operation registry when applying a stencil.laplacian
node from the IR.
*/
func (cudaPhysics *CUDAPhysicsOps) Laplacian(shape []int, invH2 float64, src []float64) ([]float64, error) {
	switch len(shape) {
	case 1:
		return cudaPhysics.Laplacian1D(src, shape[0], invH2)
	case 2:
		return cudaPhysics.Laplacian2D(src, shape[0], shape[1], invH2)
	case 3:
		return cudaPhysics.Laplacian3D(src, shape[0], shape[1], shape[2], invH2)
	default:
		return nil, fmt.Errorf("cuda_laplacian: input rank must be 1, 2, or 3 (got %d)", len(shape))
	}
}
