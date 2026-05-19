//go:build arm64

package neon

//go:noescape
func MseSumNEONAsm(predictions, targets *float32, n int) float32

//go:noescape
func MaeSumNEONAsm(predictions, targets *float32, n int) float32

//go:noescape
func L1NormNEONAsm(src *float32, n int) float32
