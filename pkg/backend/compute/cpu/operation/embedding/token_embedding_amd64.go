//go:build amd64

package embedding

import "golang.org/x/sys/cpu"

var useAVX2 bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
}

//go:noescape
func tokenEmbeddingCopyAVX2(dst, src []float64)

//go:noescape
func tokenEmbeddingCopySSE2(dst, src []float64)

func tokenEmbeddingKernel(out []float64, tokens []float64, weight []float64, dModel int) {
	for tokenIndex, token := range tokens {
		tokenID := int(token)
		dst := out[tokenIndex*dModel : (tokenIndex+1)*dModel]
		src := weight[tokenID*dModel : (tokenID+1)*dModel]

		if useAVX2 {
			tokenEmbeddingCopyAVX2(dst, src)
		} else {
			tokenEmbeddingCopySSE2(dst, src)
		}
	}
}
