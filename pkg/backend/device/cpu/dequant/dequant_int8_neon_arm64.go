//go:build arm64

package dequant

//go:noescape
func DequantInt8NEONAsm(dst *float32, src *int8, n int, scale float32, zeroPoint int16)
