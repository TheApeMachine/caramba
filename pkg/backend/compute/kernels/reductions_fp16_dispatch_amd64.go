//go:build amd64

package kernels

import "github.com/theapemachine/caramba/pkg/dtype"

func sumFloat16Native(values []dtype.F16) dtype.F16 {
	var sum float32

	for index := range values {
		sum += values[index].Float32()
	}

	return dtype.Fromfloat32(sum)
}
