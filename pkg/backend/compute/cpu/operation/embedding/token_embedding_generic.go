//go:build !amd64 && !arm64

package embedding

func tokenEmbeddingKernel(out []float64, tokens []float64, weight []float64, dModel int) {
	for tokenIndex, token := range tokens {
		tokenID := int(token)
		copy(
			out[tokenIndex*dModel:(tokenIndex+1)*dModel],
			weight[tokenID*dModel:(tokenID+1)*dModel],
		)
	}
}
