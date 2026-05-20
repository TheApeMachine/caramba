package tokenizer

/*
PackInt32Native copies src into dst using the best available CPU kernel.
*/
func PackInt32Native(dst, src []int32) {
	elementCount := len(src)

	if elementCount == 0 {
		return
	}

	packInt32Kernel(&dst[0], &src[0], elementCount)
}
