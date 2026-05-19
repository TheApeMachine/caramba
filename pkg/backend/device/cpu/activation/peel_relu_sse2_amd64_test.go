//go:build amd64

package activation

import (
	"fmt"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func TestReLUF32SSE2PeelParity(t *testing.T) {
	if !cpu.X86.HasSSE2 {
		t.Skip("SSE2 not supported")
	}

	for _, length := range parity.Lengths {
		t.Run(fmt.Sprintf("N=%d", length), func(t *testing.T) {
			source := make([]float32, length)
			simdOutput := make([]float32, length)
			reference := make([]float32, length)

			for index := range source {
				source[index] = float32(index)*0.13 - float32(length)/2
			}

			ReLUF32Generic(&reference[0], &source[0], length)
			ReLUF32SSE2(&simdOutput[0], &source[0], length)

			parity.AssertFloat32SlicesWithinULP(t, simdOutput, reference, 2)
		})
	}
}

func BenchmarkReLUF32SSE2Peel(b *testing.B) {
	if !cpu.X86.HasSSE2 {
		b.Skip("SSE2 not supported")
	}

	source := make([]float32, 8192)
	destination := make([]float32, 8192)

	for index := range source {
		source[index] = float32(index) * 0.01
	}

	for b.Loop() {
		ReLUF32SSE2(&destination[0], &source[0], len(source))
	}
}
