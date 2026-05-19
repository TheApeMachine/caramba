package activation

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &hardGeluF16LUT, &hardGeluBF16LUT, math.FastHardGelu32)
}

func QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &quickGeluF16LUT, &quickGeluBF16LUT, math.FastQuickGelu32)
}

func TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &tanhShrinkF16LUT, &tanhShrinkBF16LUT, math.FastTanhShrink32)
}
