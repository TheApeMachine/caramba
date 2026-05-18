//go:build arm64

package kernels

//go:noescape
func ropePairsNEONAsm(out, in, cos, sin *float32, pairs int)
