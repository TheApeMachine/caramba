package activation

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func Exp(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &expF16LUT, &expBF16LUT, math.FastExp32)
}

func Log(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &logF16LUT, &logBF16LUT, math.FastLog32)
}

func Log1p(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &log1pF16LUT, &log1pBF16LUT, math.FastLog1p32)
}

func Expm1(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &expm1F16LUT, &expm1BF16LUT, math.FastExpm1_32)
}

func Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &sigmoidF16LUT, &sigmoidBF16LUT, math.FastSigmoid32)
}

func LogSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &logSigmoidF16LUT, &logSigmoidBF16LUT, math.FastLogSigmoid32)
}

func Tanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &tanhF16LUT, &tanhBF16LUT, math.FastTanh32)
}

func Silu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &siluF16LUT, &siluBF16LUT, math.FastSilu32)
}

func Swish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &siluF16LUT, &siluBF16LUT, math.FastSilu32)
}

func GeluTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &geluTanhF16LUT, &geluTanhBF16LUT, math.FastGeluTanh32)
}

func Gelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &geluF16LUT, &geluBF16LUT, math.FastGelu32)
}

func ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &reluF16LUT, &reluBF16LUT, math.FastReLU32)
}

func LeakyReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &leakyReluF16LUT, &leakyReluBF16LUT, math.FastLeakyReLU32)
}

func ELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &eluF16LUT, &eluBF16LUT, math.FastELU32)
}

func CELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &celuF16LUT, &celuBF16LUT, math.FastCELU32)
}

func SELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &seluF16LUT, &seluBF16LUT, math.FastSELU32)
}

func Softplus(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &softplusF16LUT, &softplusBF16LUT, math.FastSoftplus32)
}

func Mish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &mishF16LUT, &mishBF16LUT, math.FastMish32)
}

func Softsign(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &softsignF16LUT, &softsignBF16LUT, math.FastSoftsign32)
}

func HardSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &hardSigmoidF16LUT, &hardSigmoidBF16LUT, math.FastHardSigmoid32)
}

func HardSwish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &hardSwishF16LUT, &hardSwishBF16LUT, math.FastHardSwish32)
}

func HardTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &hardTanhF16LUT, &hardTanhBF16LUT, math.FastHardTanh32)
}
