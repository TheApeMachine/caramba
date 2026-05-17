//go:build cgo && xla

package xla

// #include <stdlib.h>
// #include "xla_physics.h"
import "C"

import (
	"fmt"
	"unsafe"
)

/*
XLAPhysicsOps dispatches physics-stencil kernels (periodic 1D / 2D / 3D
Laplacian) to the XLA runtime via PJRT. The StableHLO module for each
shape is compiled once and cached inside the C++ side, keyed by tensor
shape; inv_h2 is plumbed as a runtime scalar input so the cache is shape-
indexed, not value-indexed.
*/
type XLAPhysicsOps struct {
	platform string
}

/*
NewPhysics initializes the PJRT client for physics kernels on the given
platform ("cpu" / "gpu"). Repeated calls without Shutdown leak resources.
*/
func NewPhysics(platform string) (*XLAPhysicsOps, error) {
	config, err := newRuntimePJRTConfig(platform)

	if err != nil {
		return nil, err
	}

	cp := C.CString(config.Platform)
	defer C.free(unsafe.Pointer(cp))

	if rc := C.xla_physics_init(cp); rc != 0 {
		return nil, fmt.Errorf("xla_physics_init failed for platform %q", config.Platform)
	}

	return &XLAPhysicsOps{platform: config.Platform}, nil
}

/*
Shutdown releases all PJRT physics resources.
*/
func (xlaPhysics *XLAPhysicsOps) Shutdown() { C.xla_physics_shutdown() }

/*
Laplacian1D computes the periodic 1D Laplacian over a length-n grid.
For n == 1 the periodic wrap collapses; we shortcut to a zero output and
do not invoke XLA, since the StableHLO slice operations require n >= 2.
*/
func (xlaPhysics *XLAPhysicsOps) Laplacian1D(src []float64, n int, invH2 float64) ([]float64, error) {
	if n <= 0 {
		return nil, fmt.Errorf("xla_laplacian_1d: non-positive n=%d", n)
	}

	if len(src) != n {
		return nil, fmt.Errorf("xla_laplacian_1d: src length %d != n %d", len(src), n)
	}

	dst := make([]float64, n)

	if n == 1 {
		return dst, nil
	}

	rc := C.xla_laplacian_1d(
		(*C.double)(unsafe.Pointer(&src[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(n),
		C.double(invH2),
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_laplacian_1d failed (rc=%d)", rc)
	}

	return dst, nil
}

/*
Laplacian2D computes the periodic 5-point Laplacian on a row-major
[H, W] grid. Requires H, W >= 2.
*/
func (xlaPhysics *XLAPhysicsOps) Laplacian2D(src []float64, h, w int, invH2 float64) ([]float64, error) {
	if h <= 0 || w <= 0 {
		return nil, fmt.Errorf("xla_laplacian_2d: non-positive shape (%d,%d)", h, w)
	}

	total := h * w

	if len(src) != total {
		return nil, fmt.Errorf("xla_laplacian_2d: src length %d != H*W %d", len(src), total)
	}

	dst := make([]float64, total)

	if h < 2 || w < 2 {
		return nil, fmt.Errorf("xla_laplacian_2d: both dimensions must be >= 2 (got %d,%d)", h, w)
	}

	rc := C.xla_laplacian_2d(
		(*C.double)(unsafe.Pointer(&src[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(h),
		C.int(w),
		C.double(invH2),
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_laplacian_2d failed (rc=%d)", rc)
	}

	return dst, nil
}

/*
Laplacian3D computes the periodic 7-point Laplacian on a row-major
[D, H, W] grid. Requires D, H, W >= 2.
*/
func (xlaPhysics *XLAPhysicsOps) Laplacian3D(src []float64, d, h, w int, invH2 float64) ([]float64, error) {
	if d <= 0 || h <= 0 || w <= 0 {
		return nil, fmt.Errorf("xla_laplacian_3d: non-positive shape (%d,%d,%d)", d, h, w)
	}

	total := d * h * w

	if len(src) != total {
		return nil, fmt.Errorf("xla_laplacian_3d: src length %d != D*H*W %d", len(src), total)
	}

	if d < 2 || h < 2 || w < 2 {
		return nil, fmt.Errorf("xla_laplacian_3d: all dimensions must be >= 2 (got %d,%d,%d)", d, h, w)
	}

	dst := make([]float64, total)

	rc := C.xla_laplacian_3d(
		(*C.double)(unsafe.Pointer(&src[0])),
		(*C.double)(unsafe.Pointer(&dst[0])),
		C.int(d),
		C.int(h),
		C.int(w),
		C.double(invH2),
	)

	if rc != 0 {
		return nil, fmt.Errorf("xla_laplacian_3d failed (rc=%d)", rc)
	}

	return dst, nil
}

/*
Laplacian dispatches to the rank-specific kernel based on shape length.
*/
func (xlaPhysics *XLAPhysicsOps) Laplacian(shape []int, invH2 float64, src []float64) ([]float64, error) {
	switch len(shape) {
	case 1:
		return xlaPhysics.Laplacian1D(src, shape[0], invH2)
	case 2:
		return xlaPhysics.Laplacian2D(src, shape[0], shape[1], invH2)
	case 3:
		return xlaPhysics.Laplacian3D(src, shape[0], shape[1], shape[2], invH2)
	default:
		return nil, fmt.Errorf("xla_laplacian: input rank must be 1, 2, or 3 (got %d)", len(shape))
	}
}
