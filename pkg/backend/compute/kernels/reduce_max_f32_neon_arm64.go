//go:build arm64

package kernels

//go:noescape
func reduceMaxFloat32NEONAsm(src *float32, n int) float32
