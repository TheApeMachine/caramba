package rope

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/convert"
	"github.com/theapemachine/manifesto/dtype"
)

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
