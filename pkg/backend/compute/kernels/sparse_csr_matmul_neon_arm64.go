//go:build arm64

package kernels

//go:noescape
func sparseCSRMatMulRowSingleNzNEONAsm(
	outRow *float32,
	value float32,
	denseRow *float32,
	cols int,
)
