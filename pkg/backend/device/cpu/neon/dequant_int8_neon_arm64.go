//go:build arm64

package neon

//go:noescape
func DequantInt8NEONAsm(dst *float32, src *int8, n int, scale float32, zeroPoint int16)
