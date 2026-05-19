//go:build arm64

package cpu

/*
ARM64 dispatchers for elementwise float32 add/sub/mul. The NEON
assembly bodies are in elementwise_f32_neon_arm64.s. Each call must be
preceded by length validation in the caller; the assembly does not
range-check.
*/

func AddFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	AddFloat32NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}

func SubFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	SubFloat32NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}

func MulFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	MulFloat32NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}

func DivFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	DivFloat32NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}

func MaxFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	MaxFloat32NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}

func MinFloat32Native(dst, left, right []float32) {
	if len(dst) == 0 {
		return
	}

	MinFloat32NEONAsm(&dst[0], &left[0], &right[0], len(dst))
}
