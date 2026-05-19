//go:build arm64

package cpu

import "github.com/theapemachine/caramba/pkg/dtype"

func DotBFloat16Native(a, b []dtype.BF16) dtype.BF16 {
	if len(a) == 0 {
		return 0
	}

	return dtype.BF16(DotBFloat16NEONAsm((*uint16)(&a[0]), (*uint16)(&b[0]), len(a)))
}
