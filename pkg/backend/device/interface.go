package device

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

type PosPop interface {
	CountString(counts *[8]int, str string)
	Count8(counts *[8]int, buf []uint8)
	Count16(counts *[16]int, buf []uint16)
	Count32(counts *[32]int, buf []uint32)
	Count64(counts *[64]int, buf []uint64)
}

type Activation interface {
	Exp(dst, src unsafe.Pointer, count int, format dtype.DType)
	Log(dst, src unsafe.Pointer, count int, format dtype.DType)
	Log1p(dst, src unsafe.Pointer, count int, format dtype.DType)
	Expm1(dst, src unsafe.Pointer, count int, format dtype.DType)
	Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType)
	LogSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType)
	Tanh(dst, src unsafe.Pointer, count int, format dtype.DType)
	Silu(dst, src unsafe.Pointer, count int, format dtype.DType)
	Swish(dst, src unsafe.Pointer, count int, format dtype.DType)
	GeluTanh(dst, src unsafe.Pointer, count int, format dtype.DType)
	Gelu(dst, src unsafe.Pointer, count int, format dtype.DType)
	ReLU(dst, src unsafe.Pointer, count int, format dtype.DType)
	LeakyReLU(dst, src unsafe.Pointer, count int, format dtype.DType)
	ELU(dst, src unsafe.Pointer, count int, format dtype.DType)
	CELU(dst, src unsafe.Pointer, count int, format dtype.DType)
	SELU(dst, src unsafe.Pointer, count int, format dtype.DType)
	Softplus(dst, src unsafe.Pointer, count int, format dtype.DType)
	Mish(dst, src unsafe.Pointer, count int, format dtype.DType)
	Softsign(dst, src unsafe.Pointer, count int, format dtype.DType)
	HardSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType)
	HardSwish(dst, src unsafe.Pointer, count int, format dtype.DType)
	HardTanh(dst, src unsafe.Pointer, count int, format dtype.DType)
	HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType)
	QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType)
	TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType)

	Softmax(dst, src unsafe.Pointer, count int, format dtype.DType)
	LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType)

	PReLU(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32)
	PReLUV(dst, src, slopes unsafe.Pointer, count int, format dtype.DType, slopeCount int)
	LeakyReLUSlope(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32)
	ELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32)
	CELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32)
	Threshold(dst, src unsafe.Pointer, count int, format dtype.DType, threshold float32)
	HardTanhRange(dst, src unsafe.Pointer, count int, format dtype.DType, minVal, maxVal float32)
	Snake(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32)
	SnakeParametric(dst, src unsafe.Pointer, count int, format dtype.DType, alpha, beta float32)
	HardShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32)
	SoftShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32)
	RReLU(dst, src unsafe.Pointer, count int, format dtype.DType, lower, upper float32)

	GLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	GeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	GeGLUTanh(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	SwiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	ReGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	SiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	GLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	GeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	ReGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	SiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	LinGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	SeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
	LinGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
	SeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
}
