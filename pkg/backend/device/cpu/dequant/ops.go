package dequant

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/backend/device"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func Dequant(
	dst, src unsafe.Pointer,
	count int,
	config device.DequantInt8Config,
	dstFormat, srcFormat dtype.DType,
) {
	if count == 0 {
		return
	}

	if dstFormat != dtype.Float32 || srcFormat != dtype.Int8 {
		panic("dequant: Dequant unsupported dtype pair")
	}

	if dst == nil {
		panic("dequant: nil dst pointer")
	}

	if src == nil {
		panic("dequant: nil src pointer")
	}

	dstView := unsafe.Slice((*float32)(dst), count)
	srcView := unsafe.Slice((*int8)(src), count)

	DequantInt8Native(dstView, srcView, config.Scale, config.ZeroPoint)
}

func Dequant4(
	dst, src unsafe.Pointer,
	pairCount int,
	config device.DequantInt4Config,
	dstFormat, srcFormat dtype.DType,
) {
	if pairCount == 0 {
		return
	}

	if dstFormat != dtype.Float32 || srcFormat != dtype.Int4 {
		panic("dequant: Dequant4 unsupported dtype pair")
	}

	byteCount := (pairCount + 1) / 2
	pairs := tensor.NewInt4Vector(
		unsafe.Slice((*dtype.Int4Pair)(src), byteCount),
		pairCount,
	)
	dstView := unsafe.Slice((*float32)(dst), pairCount)

	DequantInt4Native(dstView, pairs, config.Scale, config.ZeroPoint)
}
