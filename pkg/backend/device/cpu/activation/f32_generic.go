package activation

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
)

func ExpF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastExp32(source[index])
	}
}

func LogF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastLog32(source[index])
	}
}

func Log1pF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastLog1p32(source[index])
	}
}

func Expm1F32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastExpm1_32(source[index])
	}
}

func SigmoidF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastSigmoid32(source[index])
	}
}

func LogSigmoidF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastLogSigmoid32(source[index])
	}
}

func TanhF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastTanh32(source[index])
	}
}

func SiluF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastSilu32(source[index])
	}
}

func GeluTanhF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastGeluTanh32(source[index])
	}
}

func GeluF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastGelu32(source[index])
	}
}

func ReLUF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastReLU32(source[index])
	}
}

func LeakyReLUF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastLeakyReLU32(source[index])
	}
}

func ELUF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastELU32(source[index])
	}
}

func CELUF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastCELU32(source[index])
	}
}

func SELUF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastSELU32(source[index])
	}
}

func SoftplusF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastSoftplus32(source[index])
	}
}

func MishF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastMish32(source[index])
	}
}

func SoftsignF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastSoftsign32(source[index])
	}
}

func HardSigmoidF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastHardSigmoid32(source[index])
	}
}

func HardSwishF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastHardSwish32(source[index])
	}
}

func HardTanhF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastHardTanh32(source[index])
	}
}

func HardGeluF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastHardGelu32(source[index])
	}
}

func QuickGeluF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastQuickGelu32(source[index])
	}
}

func TanhShrinkF32Generic(dst, src *float32, count int) {
	destination := unsafe.Slice(dst, count)
	source := unsafe.Slice(src, count)

	for index := 0; index < count; index++ {
		destination[index] = math.FastTanhShrink32(source[index])
	}
}
