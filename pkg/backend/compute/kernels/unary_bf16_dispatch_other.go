//go:build !amd64 && !arm64

package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func absBFloat16Native(dst, src []dtype.BF16) {
	for index, value := range src {
		dst[index] = dtype.NewBfloat16FromFloat32(float32(math.Abs(float64((&value).Float32()))))
	}
}

func negBFloat16Native(dst, src []dtype.BF16) {
	for index, value := range src {
		dst[index] = dtype.NewBfloat16FromFloat32(-(&value).Float32())
	}
}

func sqrtBFloat16Native(dst, src []dtype.BF16) {
	for index, value := range src {
		dst[index] = dtype.NewBfloat16FromFloat32(float32(math.Sqrt(float64((&value).Float32()))))
	}
}

func reluBFloat16Native(dst, src []dtype.BF16) {
	for index, value := range src {
		valueF32 := (&value).Float32()
		dst[index] = dtype.NewBfloat16FromFloat32(0)

		if valueF32 > 0 {
			dst[index] = dtype.NewBfloat16FromFloat32(valueF32)
		}
	}
}
