//go:build arm64

package kernels

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/convert"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Bulk widen / narrow helpers for bf16 <-> f32 used by mixed-dtype
kernels (matmul, etc.). They route to the already-verified NEON
conversion in pkg/backend/compute/convert.
*/

func bfloat16BulkToFloat32(dst []float32, src []dtype.BF16) {
	if len(src) == 0 {
		return
	}

	_ = convert.BFloat16ToFloat32(dst, src)
}

func float32BulkToBFloat16(dst []dtype.BF16, src []float32) {
	if len(src) == 0 {
		return
	}

	_ = convert.Float32ToBFloat16(dst, src)
}
