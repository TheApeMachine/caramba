//go:build arm64

package neon

//go:noescape
func Conv3dPatchDotNEONAsm(weight, patch *float32, patchLength int) float32
