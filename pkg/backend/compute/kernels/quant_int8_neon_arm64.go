//go:build arm64

package kernels

//go:noescape
func quantInt8NEONAsm(dst *int8, src *float32, n int, invScale float32, zeroPoint int32)
