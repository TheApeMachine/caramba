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
	dispatchGatedPacked(dst, packed, batch, halfCount, format, math.FastGLU32)
}

func GeGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, math.FastGeGLU32)
}

func GeGLUTanh(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, math.FastGeGLUTanh32)
}

func SwiGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, math.FastSwiGLU32)
}

func ReGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, math.FastReGLU32)
}

func SiGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, math.FastSiGLU32)
}

func GLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, math.FastGLU32)
}

func GeGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, math.FastGeGLU32)
}

func SwiGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, math.FastSwiGLU32)
}

func ReGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, math.FastReGLU32)
}

func SiGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, math.FastSiGLU32)
}

func LinGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, math.FastLinGLU32)
}

func SeGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, math.FastSeGLU32)
}

func LinGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, math.FastLinGLU32)
}

func SeGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, math.FastSeGLU32)
}
