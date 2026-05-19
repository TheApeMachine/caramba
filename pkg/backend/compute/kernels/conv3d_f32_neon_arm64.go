//go:build arm64

package kernels

//go:noescape
func conv3dPatchDotNEONAsm(weight, patch *float32, patchLength int) float32
