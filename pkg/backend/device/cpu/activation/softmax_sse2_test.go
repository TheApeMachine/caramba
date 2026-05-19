//go:build amd64

package activation

import (
	"math/rand"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func TestSoftmaxF32SSE2(t *testing.T) {
	if !cpu.X86.HasSSE2 {
		t.Skip("SSE2 not supported")
	}

	for _, count := range parity.Lengths {
		source := make([]float32, count)
		sseOutput := make([]float32, count)
		genericOutput := make([]float32, count)

		for index := range source {
			source[index] = rand.Float32()*2 - 1
		}

		SoftmaxF32SSE2(&sseOutput[0], &source[0], count)
		SoftmaxF32Generic(&genericOutput[0], &source[0], count)

		parity.AssertFloat32SlicesWithinULP(t, sseOutput, genericOutput, 2)
	}
}

func BenchmarkSoftmaxF32SSE2(b *testing.B) {
	if !cpu.X86.HasSSE2 {
		b.Skip("SSE2 not supported")
	}

	count := 1024
	source := make([]float32, count)
	destination := make([]float32, count)

	for index := range source {
		source[index] = rand.Float32()
	}

	for b.Loop() {
		SoftmaxF32SSE2(&destination[0], &source[0], count)
	}
}

func BenchmarkSoftmaxF32Generic(b *testing.B) {
	count := 1024
	source := make([]float32, count)
	destination := make([]float32, count)

	for index := range source {
		source[index] = rand.Float32()
	}

	for b.Loop() {
		SoftmaxF32Generic(&destination[0], &source[0], count)
	}
}
