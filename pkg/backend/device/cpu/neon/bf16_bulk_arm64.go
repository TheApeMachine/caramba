//go:build arm64

package neon

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/convert"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Bulk widen / narrow helpers for bf16 <-> f32 used by mixed-dtype
kernels (matmul, etc.). They route to the already-verified NEON
conversion in pkg/backend/compute/convert.
*/

func Bfloat16BulkToFloat32(dst []float32, src []dtype.BF16) {
	if len(src) == 0 {
		return
	}

	_ = convert.BFloat16ToFloat32(dst, src)
}

func Float32BulkToBFloat16(dst []dtype.BF16, src []float32) {
	if len(src) == 0 {
		return
	}

	_ = convert.Float32ToBFloat16(dst, src)
}
