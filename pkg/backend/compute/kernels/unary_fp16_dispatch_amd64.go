//go:build amd64

package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func absFloat16Native(dst, src []dtype.F16) {
	for index, value := range src {
		dst[index] = dtype.Fromfloat32(float32(math.Abs(float64(value.Float32()))))
	}
}

func negFloat16Native(dst, src []dtype.F16) {
	for index, value := range src {
		dst[index] = dtype.Fromfloat32(-value.Float32())
	}
}

func sqrtFloat16Native(dst, src []dtype.F16) {
	for index, value := range src {
		dst[index] = dtype.Fromfloat32(float32(math.Sqrt(float64(value.Float32()))))
	}
}

func reluFloat16Native(dst, src []dtype.F16) {
	for index, value := range src {
		valueF32 := value.Float32()
		dst[index] = dtype.Fromfloat32(0)

		if valueF32 > 0 {
			dst[index] = dtype.Fromfloat32(valueF32)
		}
	}
}
