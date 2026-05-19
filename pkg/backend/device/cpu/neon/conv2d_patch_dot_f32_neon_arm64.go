//go:build arm64

package neon

//go:noescape
func Conv2dPatchDotNEONAsm(weight, patch *float32, n int) float32
