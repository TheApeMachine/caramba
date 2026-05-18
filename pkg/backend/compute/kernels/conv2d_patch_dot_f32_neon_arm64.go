//go:build arm64

package kernels

//go:noescape
func conv2dPatchDotNEONAsm(weight, patch *float32, n int) float32
