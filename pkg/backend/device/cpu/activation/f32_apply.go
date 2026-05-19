package activation

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func runExpF32(dst, src unsafe.Pointer, count int) {
	expF32Kernel((*float32)(dst), (*float32)(src), count)
}

func runLogF32(dst, src unsafe.Pointer, count int) {
	logF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runLog1pF32(dst, src unsafe.Pointer, count int) {
	log1pF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runExpm1F32(dst, src unsafe.Pointer, count int) {
	expm1F32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runSigmoidF32(dst, src unsafe.Pointer, count int) {
	sigmoidF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runLogSigmoidF32(dst, src unsafe.Pointer, count int) {
	logSigmoidF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runTanhF32(dst, src unsafe.Pointer, count int) {
	tanhF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runSiluF32(dst, src unsafe.Pointer, count int) {
	siluF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runGeluTanhF32(dst, src unsafe.Pointer, count int) {
	geluTanhF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runGeluF32(dst, src unsafe.Pointer, count int) {
	geluF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runReLUF32(dst, src unsafe.Pointer, count int) {
	reluF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runLeakyReLUF32(dst, src unsafe.Pointer, count int) {
	leakyReluF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runELUF32(dst, src unsafe.Pointer, count int) {
	eluF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runCELUF32(dst, src unsafe.Pointer, count int) {
	celuF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runSELUF32(dst, src unsafe.Pointer, count int) {
	seluF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runSoftplusF32(dst, src unsafe.Pointer, count int) {
	softplusF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runMishF32(dst, src unsafe.Pointer, count int) {
	mishF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runSoftsignF32(dst, src unsafe.Pointer, count int) {
	softsignF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runHardSigmoidF32(dst, src unsafe.Pointer, count int) {
	hardSigmoidF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runHardSwishF32(dst, src unsafe.Pointer, count int) {
	hardSwishF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runHardTanhF32(dst, src unsafe.Pointer, count int) {
	hardTanhF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runHardGeluF32(dst, src unsafe.Pointer, count int) {
	hardGeluF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runQuickGeluF32(dst, src unsafe.Pointer, count int) {
	quickGeluF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func runTanhShrinkF32(dst, src unsafe.Pointer, count int) {
	tanhShrinkF32Kernel(
		(*float32)(dst), (*float32)(src), count)
}

func dispatchActivation(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	f16LUT, bf16LUT *[65536]uint16,
	f32 func(dst, src unsafe.Pointer, count int),
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
		f32(dst, src, count)
	}
}
