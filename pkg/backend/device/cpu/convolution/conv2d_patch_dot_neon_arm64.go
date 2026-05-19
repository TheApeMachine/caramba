//go:build arm64

package convolution

//go:noescape
func Conv2dPatchDotNEONAsm(weight, patch *float32, length int) float32
