//go:build amd64

package embedding

import "golang.org/x/sys/cpu"

var useAVX2 bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
}

// CopyRowAVX2 copies len(src) float64 values from src to dst using AVX2
// 256-bit stores (4 float64 per iteration).
//
//go:noescape
func CopyRowAVX2(dst, src []float64)

// CopyRowSSE2 copies len(src) float64 values from src to dst using SSE2
// 128-bit stores (2 float64 per iteration).
//
//go:noescape
func CopyRowSSE2(dst, src []float64)

// applyLookup iterates over every token index and copies the corresponding
// weight row into out, using the best available SIMD copy.
func applyLookup(out []float64, tokens []float64, weight []float64, dModel int) {
	n := len(tokens)
	if useAVX2 {
		for t := 0; t < n; t++ {
			id := int(tokens[t])
			CopyRowAVX2(out[t*dModel:(t+1)*dModel], weight[id*dModel:(id+1)*dModel])
		}
	} else {
		for t := 0; t < n; t++ {
			id := int(tokens[t])
			CopyRowSSE2(out[t*dModel:(t+1)*dModel], weight[id*dModel:(id+1)*dModel])
		}
	}
}
