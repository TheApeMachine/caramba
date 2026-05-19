//go:build !amd64 && !arm64

package cpu

import (
	"math"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func AbsFloat16Native(dst, src []dtype.F16) {
	for index, value := range src {
		dst[index] = dtype.Fromfloat32(float32(math.Abs(float64(value.Float32()))))
	}
}

func NegFloat16Native(dst, src []dtype.F16) {
	for index, value := range src {
		dst[index] = dtype.Fromfloat32(-value.Float32())
	}
}

func SqrtFloat16Native(dst, src []dtype.F16) {
	for index, value := range src {
		dst[index] = dtype.Fromfloat32(float32(math.Sqrt(float64(value.Float32()))))
	}
}

func ReluFloat16Native(dst, src []dtype.F16) {
	for index, value := range src {
		valueF32 := value.Float32()
		dst[index] = dtype.Fromfloat32(0)

		if valueF32 > 0 {
			dst[index] = dtype.Fromfloat32(valueF32)
		}
	}
}
