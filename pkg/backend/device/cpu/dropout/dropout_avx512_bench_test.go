//go:build amd64

package dropout

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkDropoutF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	source := randomDropoutFloat32Slice(8192, 1)
	destination := make([]float32, len(source))
	seedState := DropoutSeedState(0xC0FFEE)
	keepProb := float32(0.75)
	length := len(source)

	b.SetBytes(int64(length * 4 * 2))
	b.ResetTimer()

	for b.Loop() {
		DropoutF32AVX512(&destination[0], &source[0], length, &seedState, keepProb)
	}
}

func BenchmarkDropoutF32Generic(b *testing.B) {
	source := randomDropoutFloat32Slice(8192, 2)
	destination := make([]float32, len(source))
	seedState := DropoutSeedState(0xC0FFEE)
	keepProb := float32(0.75)
	length := len(source)

	b.SetBytes(int64(length * 4 * 2))
	b.ResetTimer()

	for b.Loop() {
		DropoutF32Generic(&destination[0], &source[0], length, &seedState, keepProb)
	}
}
