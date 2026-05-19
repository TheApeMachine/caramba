package layernorm

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var parityNs = []int{1, 7, 64, 1024, 8192}

func softmaxDTypeOutputBits(out tensor.Tensor, storageDType dtype.DType) []uint16 {
	switch storageDType {
	case dtype.Float16:
		view, err := out.Float16Native()

		if err != nil {
			panic(err)
		}

		bits := make([]uint16, len(view))

		for index, value := range view {
			bits[index] = value.Bits()
		}

		return bits
	case dtype.BFloat16:
		view, err := out.BFloat16Native()

		if err != nil {
			panic(err)
		}

		bits := make([]uint16, len(view))

		for index, value := range view {
			bits[index] = value.Bits()
		}

		return bits
	default:
		panic("unsupported dtype")
	}
}

func softmaxUint16Distance(left, right uint16) uint32 {
	if left > right {
		left, right = right, left
	}

	return uint32(right - left)
}
