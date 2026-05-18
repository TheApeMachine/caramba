//go:build arm64

package kernels

import "github.com/theapemachine/caramba/pkg/dtype"

/*
ARM64 dispatchers for bf16 elementwise binary ops. The asm bodies are
in elementwise_bf16_neon_arm64.s and process 16 bf16 lanes per inner
iteration through a fused widen-compute-narrow pipeline. Caller is
responsible for length validation; the asm does not range-check.
*/

func addBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	addBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func subBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	subBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func mulBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	mulBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func divBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	divBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func maxBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	maxBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}

func minBFloat16Native(dst, left, right []dtype.BF16) {
	if len(dst) == 0 {
		return
	}

	minBFloat16NEONAsm((*uint16)(&dst[0]), (*uint16)(&left[0]), (*uint16)(&right[0]), len(dst))
}
