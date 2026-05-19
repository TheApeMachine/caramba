//go:build arm64

package neon

//go:noescape
func SparseCSRMatMulRowSingleNzNEONAsm(
	outRow *float32,
	value float32,
	denseRow *float32,
	cols int,
)
