package neon

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/convert"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Bulk widen / narrow helpers for fp16 <-> f32 used by mixed-dtype
matmul and other kernels that compute in f32. The conversion package
routes to NEON FCVTL/FCVTN on arm64 and scalar elsewhere.
*/

func Float16BulkToFloat32(dst []float32, src []dtype.F16) {
	if len(src) == 0 {
		return
	}

	_ = convert.Float16ToFloat32(dst, src)
}

func Float32BulkToFloat16(dst []dtype.F16, src []float32) {
	if len(src) == 0 {
		return
	}

	_ = convert.Float32ToFloat16(dst, src)
}
