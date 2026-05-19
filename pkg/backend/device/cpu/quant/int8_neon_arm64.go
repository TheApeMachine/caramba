//go:build arm64

package quant

//go:noescape
func QuantInt8NEONAsm(dst *int8, src *float32, n int, invScale float32, zeroPoint int32)
