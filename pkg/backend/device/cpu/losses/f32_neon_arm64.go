//go:build arm64

package losses

//go:noescape
func MseSumNEONAsm(predictions, targets *float32, count int) float32

//go:noescape
func MaeSumNEONAsm(predictions, targets *float32, count int) float32
