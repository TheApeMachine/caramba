//go:build arm64

package kernels

import "github.com/theapemachine/caramba/pkg/dtype"

func sumFloat16Native(values []dtype.F16) dtype.F16 {
	if len(values) == 0 {
		return 0
	}

	return dtype.F16(sumFloat16NEONAsm((*uint16)(&values[0]), len(values)))
}
