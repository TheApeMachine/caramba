//go:build arm64

package kernels

//go:noescape
func reduceProdFloat32NEONAsm(src *float32, n int) float32
