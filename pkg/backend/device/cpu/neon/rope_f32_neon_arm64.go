//go:build arm64

package neon

//go:noescape
func RopePairsNEONAsm(out, in, cos, sin *float32, pairs int)
