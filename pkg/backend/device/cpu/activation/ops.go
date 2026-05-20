package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func Exp(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &expF16LUT, &expBF16LUT, runExpF32)
}

func Log(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &logF16LUT, &logBF16LUT, runLogF32)
}

func Log1p(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &log1pF16LUT, &log1pBF16LUT, runLog1pF32)
}

func Expm1(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &expm1F16LUT, &expm1BF16LUT, runExpm1F32)
}

func Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &sigmoidF16LUT, &sigmoidBF16LUT, runSigmoidF32)
}

func LogSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &logSigmoidF16LUT, &logSigmoidBF16LUT, runLogSigmoidF32)
}

func Tanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &tanhF16LUT, &tanhBF16LUT, runTanhF32)
}

func Silu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &siluF16LUT, &siluBF16LUT, runSiluF32)
}

func Swish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &siluF16LUT, &siluBF16LUT, runSiluF32)
}

func GeluTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &geluTanhF16LUT, &geluTanhBF16LUT, runGeluTanhF32)
}

func Gelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &geluF16LUT, &geluBF16LUT, runGeluF32)
}

func ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &reluF16LUT, &reluBF16LUT, runReLUF32)
}

func LeakyReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &leakyReluF16LUT, &leakyReluBF16LUT, runLeakyReLUF32)
}

func ELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &eluF16LUT, &eluBF16LUT, runELUF32)
}

func CELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &celuF16LUT, &celuBF16LUT, runCELUF32)
}

func SELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &seluF16LUT, &seluBF16LUT, runSELUF32)
}

func Softplus(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &softplusF16LUT, &softplusBF16LUT, runSoftplusF32)
}

func Mish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &mishF16LUT, &mishBF16LUT, runMishF32)
}

func Softsign(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &softsignF16LUT, &softsignBF16LUT, runSoftsignF32)
}

func HardSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &hardSigmoidF16LUT, &hardSigmoidBF16LUT, runHardSigmoidF32)
}

func HardSwish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &hardSwishF16LUT, &hardSwishBF16LUT, runHardSwishF32)
}

func HardTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	dispatchActivation(dst, src, count, format, &hardTanhF16LUT, &hardTanhBF16LUT, runHardTanhF32)
}
