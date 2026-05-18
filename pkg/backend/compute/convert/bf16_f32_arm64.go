//go:build arm64

package convert

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
ARM64 NEON paths for BF16↔F32. The .s file implements the widening
shift trick: bf16 is the upper 16 bits of its float32 representation,
so the conversion is a single VUSHLL #16 instruction per 4 lanes
(8 lanes at a time with the upper/lower pair).

Verified on linux/arm64 (Go 1.26).
*/

//go:noescape
func bfloat16ToFloat32NEONAsm(dst *float32, src *uint16, n int) int

//go:noescape
func float32ToBFloat16NEONAsm(dst *uint16, src *float32, n int) int

func bfloat16ToFloat32(dst []float32, src []dtype.BF16) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	if len(src) == 0 {
		return nil
	}

	bfloat16ToFloat32NEONAsm(
		&dst[0],
		(*uint16)(unsafe.Pointer(&src[0])),
		len(src),
	)

	return nil
}

func float32ToBFloat16(dst []dtype.BF16, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	if len(src) == 0 {
		return nil
	}

	float32ToBFloat16NEONAsm(
		(*uint16)(unsafe.Pointer(&dst[0])),
		&src[0],
		len(src),
	)

	return nil
}

// bfloat16ToFloat32NEONLoop is retained for the
// dispatcher-with-scalar-tail pattern; the asm path handles its own
// tail so this just forwards.
func bfloat16ToFloat32NEONLoop(dst []float32, src []dtype.BF16) int {
	_ = bfloat16ToFloat32(dst, src)
	return len(src)
}

func float32ToBFloat16NEONLoop(dst []dtype.BF16, src []float32) int {
	_ = float32ToBFloat16(dst, src)
	return len(src)
}
