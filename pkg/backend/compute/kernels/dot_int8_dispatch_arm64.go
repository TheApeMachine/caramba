//go:build arm64

package kernels

func dotInt8Native(a, b []int8) int32 {
	if len(a) == 0 {
		return 0
	}

	return dotInt8NEONAsm(&a[0], &b[0], len(a))
}
