package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &hardGeluF16LUT, &hardGeluBF16LUT, runHardGeluF32)
}

func QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &quickGeluF16LUT, &quickGeluBF16LUT, runQuickGeluF32)
}

func TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &tanhShrinkF16LUT, &tanhShrinkBF16LUT, runTanhShrinkF32)
}
