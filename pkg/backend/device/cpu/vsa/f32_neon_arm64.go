//go:build arm64

package vsa

//go:noescape
func VsaPermuteCopyF32NEONAsm(dst, src *float32, n int)
