package activation

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
)

func SoftmaxF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)
	math.SoftmaxF32(destination, source)
}

func LogSoftmaxF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)
	math.LogSoftmaxF32(destination, source)
}
