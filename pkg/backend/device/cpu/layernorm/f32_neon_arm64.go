//go:build arm64

package layernorm

//go:noescape
func LayerNormApplyRowNEONAsm(out, row, scale, bias *float32, n int, mean, invStdDev float32)

//go:noescape
func LayerNormSquaredDiffSumNEONAsm(row *float32, n int, mean float32) float32
