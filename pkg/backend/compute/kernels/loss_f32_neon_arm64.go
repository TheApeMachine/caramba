//go:build arm64

package kernels

//go:noescape
func mseSumNEONAsm(predictions, targets *float32, n int) float32

//go:noescape
func maeSumNEONAsm(predictions, targets *float32, n int) float32

//go:noescape
func l1NormNEONAsm(src *float32, n int) float32
