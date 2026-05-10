//go:build arm64

package embedding

// CopyRowNEON copies len(src) float64 values from src to dst using NEON
// 128-bit loads/stores (2 float64 per iteration).
//
//go:noescape
func CopyRowNEON(dst, src []float64)

// applyLookup iterates over every token index and copies the corresponding
// weight row into out using NEON vector copy.
func applyLookup(out []float64, tokens []float64, weight []float64, dModel int) {
	n := len(tokens)
	for t := 0; t < n; t++ {
		id := int(tokens[t])
		CopyRowNEON(out[t*dModel:(t+1)*dModel], weight[id*dModel:(id+1)*dModel])
	}
}
