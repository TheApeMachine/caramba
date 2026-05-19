//go:build amd64

package shape

//go:noescape
func CopyContiguousFloat32AVX512Asm(dst, src *float32, count int)

//go:noescape
func WhereFloat32AVX512Asm(dst, positive, negative *float32, mask *byte, count int)

//go:noescape
func MaskedFillFloat32AVX512Asm(dst, input *float32, fill float32, mask *byte, count int)

func CopyContiguousF32AVX512(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	CopyContiguousFloat32AVX512Asm(dst, src, count)
}

func WhereF32AVX512(dst, positive, negative *float32, mask []byte, count int) {
	if count == 0 {
		return
	}

	WhereFloat32AVX512Asm(dst, positive, negative, &mask[0], count)
}

func MaskedFillF32AVX512(dst, input *float32, fill float32, mask []byte, count int) {
	if count == 0 {
		return
	}

	MaskedFillFloat32AVX512Asm(dst, input, fill, &mask[0], count)
}
