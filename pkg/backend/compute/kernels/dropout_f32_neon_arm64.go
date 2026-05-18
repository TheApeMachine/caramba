//go:build arm64

package kernels

//go:noescape
func dropoutFloat32NEONAsm(dst, src *float32, n int, seedState *uint32, scale, threshold float32)
