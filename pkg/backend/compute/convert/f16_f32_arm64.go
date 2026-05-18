//go:build arm64

package convert

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
ARM64 NEON paths for F16↔F32 using the FCVTL / FCVTN instructions
introduced in armv8-A. Single-instruction conversion per 4 lanes,
verified on linux/arm64 (Go 1.26).
*/

//go:noescape
func float16ToFloat32NEONAsm(dst *float32, src *uint16, n int) int

//go:noescape
func float32ToFloat16NEONAsm(dst *uint16, src *float32, n int) int

func float16ToFloat32Native(dst []float32, src []dtype.F16) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	if len(src) == 0 {
		return nil
	}

	float16ToFloat32NEONAsm(
		&dst[0],
		(*uint16)(unsafe.Pointer(&src[0])),
		len(src),
	)

	return nil
}

func float32ToFloat16Native(dst []dtype.F16, src []float32) error {
	if len(dst) != len(src) {
		return errLenMismatch
	}

	if len(src) == 0 {
		return nil
	}

	float32ToFloat16NEONAsm(
		(*uint16)(unsafe.Pointer(&dst[0])),
		&src[0],
		len(src),
	)

	return nil
}
