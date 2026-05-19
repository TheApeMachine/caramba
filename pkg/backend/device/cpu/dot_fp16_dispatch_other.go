//go:build !amd64 && !arm64

package cpu

import "github.com/theapemachine/caramba/pkg/dtype"

func DotFloat16Native(a, b []dtype.F16) dtype.F16 {
	var sum float32

	for index := range a {
		sum += a[index].Float32() * b[index].Float32()
	}

	return dtype.Fromfloat32(sum)
}
