//go:build amd64

package tokenizer

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkTokenizerPackInt32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const benchLen = 8192
	source := randomInt32Vector(benchLen, 0x2830)
	destination := make([]int32, benchLen)

	b.SetBytes(int64(benchLen * 4))
	b.ResetTimer()

	for b.Loop() {
		TokenizerPackInt32AVX512(&destination[0], &source[0], benchLen)
	}
}
