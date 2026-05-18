//go:build arm64

package kernels

//go:noescape
func softmaxRowExpSumNEONAsm(dst, src *float32, maxValue float32, n int) float32
