//go:build amd64

package kernels

import "github.com/theapemachine/caramba/pkg/dtype"

func dotBFloat16Native(a, b []dtype.BF16) dtype.BF16 {
	var sum float32

	for index := range a {
		sum += (&a[index]).Float32() * (&b[index]).Float32()
	}

	return dtype.NewBfloat16FromFloat32(sum)
}
