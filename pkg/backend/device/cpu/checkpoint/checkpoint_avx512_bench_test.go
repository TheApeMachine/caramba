//go:build amd64

package checkpoint

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkCheckpointEncodeFloat32DataAVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const benchLen = 8192
	source := randomFloat32Vector(benchLen, 0x2960)
	destination := make([]uint8, benchLen*4)

	b.SetBytes(int64(benchLen * 4))
	b.ResetTimer()

	for b.Loop() {
		CheckpointEncodeFloat32DataAVX512(&destination[0], &source[0], benchLen)
	}
}

func BenchmarkCheckpointDecodeFloat32DataAVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const benchLen = 8192
	source := randomFloat32Vector(benchLen, 0x2961)
	payload := make([]uint8, benchLen*4)
	encodeFloat32DataScalar(payload, source)
	destination := make([]float32, benchLen)

	b.SetBytes(int64(benchLen * 4))
	b.ResetTimer()

	for b.Loop() {
		CheckpointDecodeFloat32DataAVX512(&destination[0], &payload[0], benchLen)
	}
}
