//go:build arm64

package cpu

import "github.com/theapemachine/caramba/pkg/dtype"

func SumFloat16Native(values []dtype.F16) dtype.F16 {
	if len(values) == 0 {
		return 0
	}

	return dtype.F16(SumFloat16NEONAsm((*uint16)(&values[0]), len(values)))
}
