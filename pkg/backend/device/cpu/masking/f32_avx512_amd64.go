//go:build amd64

package masking

//go:noescape
func ApplyMaskFloat32AVX512Asm(input, mask, output *float32, count int)

//go:noescape
func CausalMaskFloat32AVX512Asm(output *float32, seqQ, seqK int)

//go:noescape
func ALiBiBiasFloat32AVX512Asm(scores, slope, output *float32, seqQ, seqK int)

func ApplyMaskF32AVX512(input, mask, output *float32, count int) {
	if count == 0 {
		return
	}

	ApplyMaskFloat32AVX512Asm(input, mask, output, count)
}

func CausalMaskF32AVX512(output *float32, seqQ, seqK int) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	CausalMaskFloat32AVX512Asm(output, seqQ, seqK)
}

func ALiBiBiasF32AVX512(scores, slope, output *float32, seqQ, seqK int) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	ALiBiBiasFloat32AVX512Asm(scores, slope, output, seqQ, seqK)
}
