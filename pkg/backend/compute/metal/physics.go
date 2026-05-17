//go:build darwin && cgo

package metal

// #include <stdlib.h>
// #include "physics.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
MetalPhysics dispatches physics-stencil kernels (1D / 2D / 3D periodic
Laplacian) to the Metal GPU. The struct caches the metallib path used at
construction so the Init pipeline state persists for the process lifetime
— the underlying global state in physics.m mirrors MetalPositional.

Metal compute uses float32 in shaders; the wrappers convert from the
caller's []float64 to []float32 on the way in and back on the way out.
On Apple Silicon this is acceptable because Metal lacks fp64 in compute
pipelines. The CPU and CUDA paths remain double-precision references and
parity tests should use a looser tolerance for Metal (~ float32 epsilon
scaled by stencil reach, e.g. 1e-5 for the 2D 5-point stencil).
*/
type MetalPhysics struct {
	metallib string
	runtime  *MetalRuntime
}

/*
NewPhysics creates and initializes a MetalPhysics. metallib must be the
absolute path to a compiled physics.metallib produced by compiling
physics.metal with `xcrun -sdk macosx metal` + `metallib`.
*/
func NewPhysics(metallib string) (*MetalPhysics, error) {
	cpath := C.CString(metallib)
	defer C.free(unsafe.Pointer(cpath))

	if rc := C.metal_physics_init(cpath); rc != 0 {
		return nil, fmt.Errorf("metal_physics_init failed (rc=%d)", rc)
	}

	runtime, err := newStandaloneMetalRuntime()

	if err != nil {
		return nil, err
	}

	return &MetalPhysics{metallib: metallib, runtime: runtime}, nil
}

/*
Laplacian1D computes the 2nd-order central-difference Laplacian on a
length-n periodic 1D grid.
*/
func (metalPhysics *MetalPhysics) Laplacian1D(src []float64, n int, invH2 float64) ([]float64, error) {
	if n <= 0 {
		return nil, fmt.Errorf("metal_laplacian_1d: non-positive n=%d", n)
	}

	if len(src) != n {
		return nil, fmt.Errorf("metal_laplacian_1d: src length %d != n %d", len(src), n)
	}

	src32 := toFloat32(src)
	dst32 := make([]float32, n)

	rc := C.metal_laplacian_1d(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(n),
		C.float(invH2),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_laplacian_1d failed (rc=%d)", rc)
	}

	return toFloat64(dst32), nil
}

/*
Laplacian2D computes the periodic 5-point Laplacian on a row-major
[H, W] grid.
*/
func (metalPhysics *MetalPhysics) Laplacian2D(src []float64, h, w int, invH2 float64) ([]float64, error) {
	if h <= 0 || w <= 0 {
		return nil, fmt.Errorf("metal_laplacian_2d: non-positive shape (%d,%d)", h, w)
	}

	total := h * w

	if len(src) != total {
		return nil, fmt.Errorf("metal_laplacian_2d: src length %d != H*W %d", len(src), total)
	}

	src32 := toFloat32(src)
	dst32 := make([]float32, total)

	rc := C.metal_laplacian_2d(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(h),
		C.int(w),
		C.float(invH2),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_laplacian_2d failed (rc=%d)", rc)
	}

	return toFloat64(dst32), nil
}

/*
Laplacian3D computes the periodic 7-point Laplacian on a row-major
[D, H, W] grid.
*/
func (metalPhysics *MetalPhysics) Laplacian3D(src []float64, d, h, w int, invH2 float64) ([]float64, error) {
	if d <= 0 || h <= 0 || w <= 0 {
		return nil, fmt.Errorf("metal_laplacian_3d: non-positive shape (%d,%d,%d)", d, h, w)
	}

	total := d * h * w

	if len(src) != total {
		return nil, fmt.Errorf("metal_laplacian_3d: src length %d != D*H*W %d", len(src), total)
	}

	src32 := toFloat32(src)
	dst32 := make([]float32, total)

	rc := C.metal_laplacian_3d(
		(*C.float)(unsafe.Pointer(&src32[0])),
		(*C.float)(unsafe.Pointer(&dst32[0])),
		C.int(d),
		C.int(h),
		C.int(w),
		C.float(invH2),
	)

	if rc != 0 {
		return nil, fmt.Errorf("metal_laplacian_3d failed (rc=%d)", rc)
	}

	return toFloat64(dst32), nil
}

/*
Laplacian dispatches to the rank-specific kernel based on shape length.
*/
func (metalPhysics *MetalPhysics) Laplacian(shape []int, invH2 float64, src []float64) ([]float64, error) {
	switch len(shape) {
	case 1:
		return metalPhysics.Laplacian1D(src, shape[0], invH2)
	case 2:
		return metalPhysics.Laplacian2D(src, shape[0], shape[1], invH2)
	case 3:
		return metalPhysics.Laplacian3D(src, shape[0], shape[1], shape[2], invH2)
	default:
		return nil, fmt.Errorf("metal_laplacian: input rank must be 1, 2, or 3 (got %d)", len(shape))
	}
}
