//go:build arm64

package kernels

import "github.com/theapemachine/caramba/pkg/dtype"

func sumBFloat16Native(values []dtype.BF16) dtype.BF16 {
	if len(values) == 0 {
		return 0
	}

	return dtype.BF16(sumBFloat16NEONAsm((*uint16)(&values[0]), len(values)))
}
