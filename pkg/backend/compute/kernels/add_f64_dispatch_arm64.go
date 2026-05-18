//go:build arm64

package kernels

func addFloat64Native(dst, left, right []float64) {
	if len(dst) == 0 {
		return
	}

	addFloat64NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}
