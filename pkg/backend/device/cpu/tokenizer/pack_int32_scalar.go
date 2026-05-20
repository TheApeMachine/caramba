package tokenizer

/*
PackInt32Scalar copies src into dst (tokenizer_pack_int32 reference).
*/
func PackInt32Scalar(dst, src []int32) {
	copy(dst, src)
}
