//go:build arm64

package neon

//go:noescape
func CounterfactualF32NEONAsm(out, observedY, observedX, counterfactualX *float32, slope float32, n int)

//go:noescape
func StridedDotF32NEONAsm(values *float32, stride int, weights *float32, n int) float32
