//go:build amd64

package attention

//go:noescape
func FlashAttentionOnlineUpdateAVX512Asm(
	acc, valueRow *float32,
	alpha, shifted float32,
	length int,
)

//go:noescape
func FlashAttentionScaleAVX512Asm(
	out, acc *float32,
	invNormalizer float32,
	length int,
)

func flashAttentionOnlineUpdateAVX512(
	acc, valueRow *float32,
	alpha, shifted float32,
	length int,
) {
	if length == 0 {
		return
	}

	FlashAttentionOnlineUpdateAVX512Asm(acc, valueRow, alpha, shifted, length)
}

func flashAttentionScaleAVX512(
	out, acc *float32,
	invNormalizer float32,
	length int,
) {
	if length == 0 {
		return
	}

	FlashAttentionScaleAVX512Asm(out, acc, invNormalizer, length)
}
