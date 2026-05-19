//go:build arm64

package attention

//go:noescape
func FlashAttentionOnlineUpdateNEONAsm(
	acc, valueRow *float32,
	alpha, shifted float32,
	n int,
)

//go:noescape
func FlashAttentionScaleNEONAsm(
	out, acc *float32,
	invNormalizer float32,
	n int,
)
