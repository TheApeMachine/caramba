//go:build arm64

package dequant

//go:noescape
func DequantInt4NEONAsm(dst *float32, src *byte, n int, scale float32, zeroPoint int8)
