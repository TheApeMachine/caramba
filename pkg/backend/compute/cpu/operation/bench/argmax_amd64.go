//go:build amd64

package bench

import "golang.org/x/sys/cpu"

//go:noescape
func argmaxAVX2(xs []float64) int

//go:noescape
func argmaxSSE2(xs []float64) int

func argmaxImpl(xs []float64) int {
	if cpu.X86.HasAVX2 {
		return argmaxAVX2(xs)
	}

	return argmaxSSE2(xs)
}
