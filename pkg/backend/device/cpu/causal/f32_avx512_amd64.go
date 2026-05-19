//go:build amd64

package causal

//go:noescape
func CateFloat32AVX512Asm(treated, control, out *float32, count int)

//go:noescape
func CounterfactualFloat32AVX512Asm(
	out, observedY, observedX, counterfactualX *float32,
	slope float32,
	count int,
)

//go:noescape
func StridedDotFloat32AVX512Asm(values *float32, stride int, weights *float32, count int) float32

func cateF32AVX512(treated, control, out []float32) {
	if len(out) == 0 {
		return
	}

	CateFloat32AVX512Asm(&treated[0], &control[0], &out[0], len(out))
}

func counterfactualF32AVX512(
	out, observedY, observedX, counterfactualX []float32,
	slope float32,
) {
	elementCount := len(out)

	if elementCount == 0 {
		return
	}

	CounterfactualFloat32AVX512Asm(
		&out[0], &observedY[0], &observedX[0], &counterfactualX[0],
		slope, elementCount,
	)
}

func stridedDotF32AVX512(values []float32, stride int, weights []float32, elementCount int) float32 {
	if elementCount == 0 {
		return 0
	}

	return StridedDotFloat32AVX512Asm(&values[0], stride, &weights[0], elementCount)
}
