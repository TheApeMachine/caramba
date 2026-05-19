//go:build amd64

package sampling

//go:noescape
func GreedySampleFloat32AVX512Asm(logits *float32, count int) int32

//go:noescape
func SamplingSoftmaxRowFloat32AVX512Asm(logits, out *float32, temperature float32, count int)

func GreedySampleF32AVX512(logits *float32, count int) int32 {
	if count == 0 {
		return 0
	}

	return GreedySampleFloat32AVX512Asm(logits, count)
}

func SamplingSoftmaxRowF32AVX512(logits, out *float32, temperature float32, count int) {
	if count == 0 {
		return
	}

	SamplingSoftmaxRowFloat32AVX512Asm(logits, out, temperature, count)
}
