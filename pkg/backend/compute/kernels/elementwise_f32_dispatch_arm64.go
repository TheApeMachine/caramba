//go:build arm64

package kernels

/*
ARM64 dispatchers for elementwise float32 add/sub/mul. The NEON
assembly bodies are in elementwise_f32_neon_arm64.s. Each call must be
preceded by length validation in the caller; the assembly does not
range-check.
*/

func addFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	addFloat32NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}

func subFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	subFloat32NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}

func mulFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	mulFloat32NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}

func divFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	divFloat32NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}

func maxFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	maxFloat32NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}

func minFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	minFloat32NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}
