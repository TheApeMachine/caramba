//go:build arm64

package kernels

//go:noescape
func dequantInt8NEONAsm(dst *float32, src *int8, n int, scale float32, zeroPoint int16)
