//go:build arm64

package rope

//go:noescape
func RopePairsNEONAsm(out, in, cos, sin *float32, pairs int)
