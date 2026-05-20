package dot

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func dispatchDot(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	if count == 0 {
		return 0
	}

	switch format {
	case dtype.Float32:
		return runDotF32(left, right, count)
	case dtype.BFloat16:
		bf16Result := dispatchDotBF16(left, right, count)
		return (&bf16Result).Float32()
	case dtype.Float16:
		fp16Result := dispatchDotFP16(left, right, count)
		return fp16Result.Float32()
	case dtype.Int8:
		return float32(dispatchDotInt8(left, right, count))
	default:
		panic("dot: Dot unsupported dtype")
	}
}
