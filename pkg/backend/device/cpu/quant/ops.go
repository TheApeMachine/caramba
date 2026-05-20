package quant

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device"
	"github.com/theapemachine/manifesto/dtype"
)

func Quant(
	dst, src unsafe.Pointer,
	count int,
	config device.DequantInt8Config,
	dstFormat, srcFormat dtype.DType,
) {
	if count == 0 {
		return
	}

	if dstFormat != dtype.Int8 || srcFormat != dtype.Float32 {
		panic("quant: Quant unsupported dtype pair")
	}

	if dst == nil {
		panic("quant: nil dst pointer")
	}

	if src == nil {
		panic("quant: nil src pointer")
	}

	dstView := unsafe.Slice((*int8)(dst), count)
	srcView := unsafe.Slice((*float32)(src), count)

	QuantInt8Native(dstView, srcView, config.Scale, config.ZeroPoint)
}
