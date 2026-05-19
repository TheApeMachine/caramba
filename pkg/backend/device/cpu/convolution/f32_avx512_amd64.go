//go:build amd64

package convolution

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:noescape
func ConvPatchDotFloat32AVX512Asm(weight, patch *float32, length int) float32

func ConvPatchDotF32AVX512(weight, patch *float32, length int) float32 {
	if length == 0 {
		return 0
	}

	return ConvPatchDotFloat32AVX512Asm(weight, patch, length)
}

func convPatchDotF32Native(weight, patch *float32, length int) float32 {
	if length == 0 {
		return 0
	}

	if cpu.X86.HasAVX512F {
		return ConvPatchDotF32AVX512(weight, patch, length)
	}

	weightSlice := unsafe.Slice(weight, length)
	patchSlice := unsafe.Slice(patch, length)

	return ConvPatchDotScalar(weightSlice, patchSlice, length)
}
