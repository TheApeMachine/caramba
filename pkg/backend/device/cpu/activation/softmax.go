package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
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
		if logSpace {
			logSoftmaxF32Kernel(
				(*float32)(dst),
				(*float32)(src),
				count,
			)
			return
		}

		softmaxF32Kernel(
			(*float32)(dst),
			(*float32)(src),
			count,
		)
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
			logSoftmaxF32Kernel(&destination[0], &scratch[0], count)
		} else {
			softmaxF32Kernel(&destination[0], &scratch[0], count)
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
