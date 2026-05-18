//go:build arm64

package kernels

//go:noescape
func flashAttentionOnlineUpdateNEONAsm(
	acc, valueRow *float32,
	alpha, shifted float32,
	n int,
)

//go:noescape
func flashAttentionScaleNEONAsm(
	out, acc *float32,
	invNormalizer float32,
	n int,
)
