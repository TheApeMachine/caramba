//go:build arm64

package kernels

import "github.com/theapemachine/caramba/pkg/dtype"

func dotFloat16Native(a, b []dtype.F16) dtype.F16 {
	if len(a) == 0 {
		return 0
	}

	return dtype.F16(dotFloat16NEONAsm((*uint16)(&a[0]), (*uint16)(&b[0]), len(a)))
}
