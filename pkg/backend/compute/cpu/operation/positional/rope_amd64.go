//go:build amd64

package positional

import "golang.org/x/sys/cpu"

var useAVX2 bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
}

//go:noescape
func RoPEAVX2(dst, src, cosTable, sinTable []float64, numPairs int)

//go:noescape
func RoPESSE2(dst, src, cosTable, sinTable []float64, numPairs int)

//go:noescape
func ropeAdvanceRowAVX2(cosCur, sinCur, cosStep, sinStep []float64)

//go:noescape
func ropeAdvanceRowSSE2(cosCur, sinCur, cosStep, sinStep []float64)

func ropeAdvanceRow(cosCur, sinCur, cosStep, sinStep []float64) {
	if useAVX2 {
		ropeAdvanceRowAVX2(cosCur, sinCur, cosStep, sinStep)
		return
	}

	ropeAdvanceRowSSE2(cosCur, sinCur, cosStep, sinStep)
}

// applyRoPE dispatches the rotation over the full tensor.
// The SIMD kernels handle one (position) slice of length headDim at a time.
func ropeKernel(dst, src, cosTable, sinTable []float64, batch, numHeads, seqLen, numPairs int) {
	headDim := numPairs * 2
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for t := 0; t < seqLen; t++ {
				offset := ((b*numHeads+h)*seqLen + t) * headDim
				xSlice := src[offset : offset+headDim]
				dSlice := dst[offset : offset+headDim]
				cosSlice := cosTable[t*numPairs : (t+1)*numPairs]
				sinSlice := sinTable[t*numPairs : (t+1)*numPairs]
				if useAVX2 {
					RoPEAVX2(dSlice, xSlice, cosSlice, sinSlice, numPairs)
				} else {
					RoPESSE2(dSlice, xSlice, cosSlice, sinSlice, numPairs)
				}
			}
		}
	}
}
