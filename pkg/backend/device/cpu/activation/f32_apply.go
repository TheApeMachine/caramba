package activation

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func applyF32(
	dst, src unsafe.Pointer,
	count int,
	apply func(float32) float32,
) {
	if count == 0 {
		return
	}

	in := unsafe.Slice((*float32)(src), count)
	out := unsafe.Slice((*float32)(dst), count)

	for index := 0; index < count; index++ {
		out[index] = apply(in[index])
	}
}

func dispatchActivation(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	f16LUT, bf16LUT *[65536]uint16,
	f32Apply func(float32) float32,
) {
	if count == 0 {
		return
	}

	switch format {
	case dtype.Float16:
		applyF16LUT(dst, src, count, f16LUT)
	case dtype.BFloat16:
		applyBF16LUT(dst, src, count, bf16LUT)
	case dtype.Float32:
		applyF32(dst, src, count, f32Apply)
	}
}
