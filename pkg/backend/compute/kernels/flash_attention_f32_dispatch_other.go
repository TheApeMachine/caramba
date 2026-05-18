//go:build !arm64

package kernels

func runFlashAttentionRowNative(
	queryView, keyView, valueView, outView []float32,
	rowIndex, seqK, depth, valueDim int,
	scale float32,
	causal bool,
) {
	runFlashAttentionRow(
		queryView, keyView, valueView, outView,
		rowIndex, seqK, depth, valueDim, scale, causal,
	)
}
