//go:build arm64

package neon

//go:noescape
func SoftmaxRowExpSumNEONAsm(dst, src *float32, maxValue float32, n int) float32
