package activation

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func Softmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchSoftmax(dst, src, count, format, false)
}

func LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchSoftmax(dst, src, count, format, true)
}

func dispatchSoftmax(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	logSpace bool,
) {
	if count == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		source := unsafe.Slice((*float32)(src), count)
		destination := unsafe.Slice((*float32)(dst), count)

		if logSpace {
			math.LogSoftmaxF32(destination, source)
			return
		}

		math.SoftmaxF32(destination, source)
	case dtype.Float16, dtype.BFloat16:
		scratch := make([]float32, count)
		destination := make([]float32, count)

		for index := 0; index < count; index++ {
			if format == dtype.Float16 {
				scratch[index] = loadF16(src, index)
				continue
			}

			scratch[index] = loadBF16(src, index)
		}

		if logSpace {
			math.LogSoftmaxF32(destination, scratch)
		} else {
			math.SoftmaxF32(destination, scratch)
		}

		for index := 0; index < count; index++ {
			if format == dtype.Float16 {
				storeF16(dst, index, destination[index])
				continue
			}

			storeBF16(dst, index, destination[index])
		}
	}
}
