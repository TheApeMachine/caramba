//go:build amd64

package tokenizer

//go:noescape
func TokenizerPackInt32AVX512Asm(dst, src *int32, count int)

/*
TokenizerPackInt32AVX512 copies count int32 lanes from src to dst.
*/
func TokenizerPackInt32AVX512(dst, src *int32, count int) {
	if count == 0 {
		return
	}

	TokenizerPackInt32AVX512Asm(dst, src, count)
}
