//go:build arm64

package cpu

func DotInt8Native(a, b []int8) int32 {
	if len(a) == 0 {
		return 0
	}

	return DotInt8NEONAsm(&a[0], &b[0], len(a))
}
