//go:build amd64

package layernorm

//go:noescape
func LayerNormSquaredDiffSumFloat32AVX512Asm(row *float32, count int, mean float32) float32

//go:noescape
func LayerNormApplyRowFloat32AVX512Asm(
	out, row, scale, bias *float32,
	count int,
	mean, invStdDev float32,
)

func layerNormSquaredDiffSumF32AVX512(row []float32, mean float32) float32 {
	if len(row) == 0 {
		return 0
	}

	return LayerNormSquaredDiffSumFloat32AVX512Asm(&row[0], len(row), mean)
}

func layerNormApplyRowF32AVX512(
	outRow, row, scale, bias []float32,
	mean, invStdDev float32,
) {
	elementCount := len(row)

	if elementCount == 0 {
		return
	}

	LayerNormApplyRowFloat32AVX512Asm(
		&outRow[0], &row[0], &scale[0], &bias[0],
		elementCount, mean, invStdDev,
	)
}
