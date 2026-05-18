//go:build arm64

package kernels

//go:noescape
func dequantInt4NEONAsm(dst *float32, src *byte, n int, scale float32, zeroPoint int8)
