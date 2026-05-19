//go:build arm64

package convolution

//go:noescape
func Conv3dPatchDotNEONAsm(weight, patch *float32, patchLength int) float32
