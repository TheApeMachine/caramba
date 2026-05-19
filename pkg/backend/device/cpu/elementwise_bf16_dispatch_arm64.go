//go:build arm64

package cpu

import "github.com/theapemachine/caramba/pkg/dtype"

/*
ARM64 dispatchers for bf16 elementwise binary ops. The asm bodies are
in elementwise_bf16_neon_arm64.s and process 16 bf16 lanes per inner
iteration through a fused widen-compute-narrow pipeline. Caller is
responsible for length validation; the asm does not range-check.
*/

func AddBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	AddBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func SubBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	SubBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func MulBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	MulBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func DivBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	DivBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func MaxBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	MaxBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func MinBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	MinBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}
