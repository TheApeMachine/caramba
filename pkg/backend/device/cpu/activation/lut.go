package activation

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/manifesto/dtype"
)

type lutSpec struct {
	f16  *[65536]uint16
	bf16 *[65536]uint16
	fill func(float64) float64
}

func init() {
	specs := []lutSpec{
		{&expF16LUT, &expBF16LUT, math.FastExp64},
		{&logF16LUT, &logBF16LUT, math.FastLog64},
		{&log1pF16LUT, &log1pBF16LUT, math.FastLog1p64},
		{&expm1F16LUT, &expm1BF16LUT, math.FastExpm1_64},
		{&sigmoidF16LUT, &sigmoidBF16LUT, math.FastSigmoid64},
		{&logSigmoidF16LUT, &logSigmoidBF16LUT, math.FastLogSigmoid64},
		{&tanhF16LUT, &tanhBF16LUT, math.FastTanh64},
		{&siluF16LUT, &siluBF16LUT, math.FastSilu64},
		{&geluTanhF16LUT, &geluTanhBF16LUT, math.FastGeluTanh64},
		{&geluF16LUT, &geluBF16LUT, math.FastGelu64},
		{&reluF16LUT, &reluBF16LUT, math.FastReLU64},
		{&leakyReluF16LUT, &leakyReluBF16LUT, math.FastLeakyReLU64},
		{&eluF16LUT, &eluBF16LUT, math.FastELU64},
		{&celuF16LUT, &celuBF16LUT, math.FastCELU64},
		{&seluF16LUT, &seluBF16LUT, math.FastSELU64},
		{&softplusF16LUT, &softplusBF16LUT, math.FastSoftplus64},
		{&mishF16LUT, &mishBF16LUT, math.FastMish64},
		{&softsignF16LUT, &softsignBF16LUT, math.FastSoftsign64},
		{&hardSigmoidF16LUT, &hardSigmoidBF16LUT, math.FastHardSigmoid64},
		{&hardSwishF16LUT, &hardSwishBF16LUT, math.FastHardSwish64},
		{&hardTanhF16LUT, &hardTanhBF16LUT, math.FastHardTanh64},
		{&hardGeluF16LUT, &hardGeluBF16LUT, math.FastHardGelu64},
		{&quickGeluF16LUT, &quickGeluBF16LUT, math.FastQuickGelu64},
		{&tanhShrinkF16LUT, &tanhShrinkBF16LUT, math.FastTanhShrink64},
	}

	for _, spec := range specs {
		fillF16LUT(spec.f16, spec.fill)
		fillBF16LUT(spec.bf16, spec.fill)
	}
}

func applyF16LUT(dst, src unsafe.Pointer, count int, lut *[65536]uint16) {
	in := unsafe.Slice((*uint16)(src), count)
	out := unsafe.Slice((*uint16)(dst), count)

	index := 0
	for index <= count-4 {
		out[index] = lut[in[index]]
		out[index+1] = lut[in[index+1]]
		out[index+2] = lut[in[index+2]]
		out[index+3] = lut[in[index+3]]
		index += 4
	}

	for index < count {
		out[index] = lut[in[index]]
		index++
	}
}

func applyBF16LUT(dst, src unsafe.Pointer, count int, lut *[65536]uint16) {
	applyF16LUT(dst, src, count, lut)
}

func fillF16LUT(lut *[65536]uint16, op func(float64) float64) {
	for index := 0; index < 65536; index++ {
		value := dtype.Frombits(uint16(index)).Float32()
		lut[index] = dtype.Fromfloat32(float32(op(float64(value)))).Bits()
	}
}

func fillBF16LUT(lut *[65536]uint16, op func(float64) float64) {
	for index := 0; index < 65536; index++ {
		bf16 := dtype.BF16(uint16(index))
		value := (&bf16).Float32()
		lut[index] = uint16(dtype.NewBfloat16FromFloat32(float32(op(float64(value)))))
	}
}
