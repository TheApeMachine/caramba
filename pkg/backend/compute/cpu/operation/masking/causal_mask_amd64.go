//go:build amd64

package masking

import "golang.org/x/sys/cpu"

var useAVX2 bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
}

//go:noescape
func CausalMaskAVX2(dst []float64, seqLen int)

//go:noescape
func CausalMaskSSE2(dst []float64, seqLen int)

func causalMaskKernel(dst []float64, seqLen int) {
	if useAVX2 {
		CausalMaskAVX2(dst, seqLen)
	} else {
		CausalMaskSSE2(dst, seqLen)
	}
}
