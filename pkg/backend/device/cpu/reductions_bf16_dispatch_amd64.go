//go:build amd64

package cpu

import "github.com/theapemachine/caramba/pkg/dtype"

func SumBFloat16Native(values []dtype.BF16) dtype.BF16 {
	var sum float32

	for index := range values {
		sum += (&values[index]).Float32()
	}

	return dtype.NewBfloat16FromFloat32(sum)
}
