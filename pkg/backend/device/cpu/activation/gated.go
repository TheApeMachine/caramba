package activation

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/math"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func GLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, gluPackedKernel, math.FastGLU32)
}

func GeGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, geGLUPackedKernel, math.FastGeGLU32)
}

func GeGLUTanh(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, geGLUTanhPackedKernel, math.FastGeGLUTanh32)
}

func SwiGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, swiGLUPackedKernel, math.FastSwiGLU32)
}

func ReGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, reGLUPackedKernel, math.FastReGLU32)
}

func SiGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, siGLUPackedKernel, math.FastSiGLU32)
}

func GLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, gluTensorsKernel, math.FastGLU32)
}

func GeGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, geGLUTensorsKernel, math.FastGeGLU32)
}

func GeGLUTanhTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, geGLUTanhTensorsKernel, math.FastGeGLUTanh32)
}

func SwiGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, swiGLUTensorsKernel, math.FastSwiGLU32)
}

func ReGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, reGLUTensorsKernel, math.FastReGLU32)
}

func SiGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, siGLUTensorsKernel, math.FastSiGLU32)
}

func LinGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, linGLUPackedKernel, math.FastLinGLU32)
}

func SeGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, seGLUPackedKernel, math.FastSeGLU32)
}

func LinGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, linGLUTensorsKernel, math.FastLinGLU32)
}

func SeGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, seGLUTensorsKernel, math.FastSeGLU32)
}
