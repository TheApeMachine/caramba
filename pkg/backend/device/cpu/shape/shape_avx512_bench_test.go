//go:build amd64

package shape

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkCopyContiguousF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	length := 8192
	source := randomShapeFloat32(length, 0x1770)
	destination := make([]float32, length)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		CopyContiguousF32AVX512(&destination[0], &source[0], length)
	}
}

func BenchmarkWhereF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	length := 8192
	positive := randomShapeFloat32(length, 0x1771)
	negative := randomShapeFloat32(length, 0x1772)
	mask := shapeMaskBytes(length, 0x1773)
	destination := make([]float32, length)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		WhereF32AVX512(&destination[0], &positive[0], &negative[0], mask, length)
	}
}

func BenchmarkMaskedFillF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	length := 8192
	input := randomShapeFloat32(length, 0x1774)
	mask := shapeMaskBytes(length, 0x1775)
	destination := make([]float32, length)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		MaskedFillF32AVX512(&destination[0], &input[0], 1.5, mask, length)
	}
}
