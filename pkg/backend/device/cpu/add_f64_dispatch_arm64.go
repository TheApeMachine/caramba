//go:build arm64

package cpu

func AddFloat64Native(dst, left, right []float64) {
	if len(dst) == 0 {
		return
	}

	AddFloat64NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}
