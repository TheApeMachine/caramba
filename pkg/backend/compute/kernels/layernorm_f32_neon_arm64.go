//go:build arm64

package kernels

//go:noescape
func layerNormApplyRowNEONAsm(out, row, scale, bias *float32, n int, mean, invStdDev float32)

//go:noescape
func layerNormSquaredDiffSumNEONAsm(row *float32, n int, mean float32) float32
