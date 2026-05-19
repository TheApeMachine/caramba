//go:build amd64

package embedding

import (
	"testing"
	"unsafe"

	"golang.org/x/sys/cpu"
)

func BenchmarkLookupF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const vocab = 8192
	const hidden = 8192
	const indexCount = 64

	table := randomEmbeddingTable(vocab, hidden, 0xE140)
	indices := randomEmbeddingIndices(indexCount, vocab, 0xE141)
	output := make([]float32, indexCount*hidden)

	b.SetBytes(int64(indexCount * hidden * 4))
	b.ResetTimer()

	for b.Loop() {
		runLookupF32AVX512(
			unsafe.Pointer(&table[0]),
			unsafe.Pointer(&indices[0]),
			unsafe.Pointer(&output[0]),
			vocab, hidden, indexCount,
		)
	}
}

func BenchmarkLookupF32Generic(b *testing.B) {
	const vocab = 8192
	const hidden = 8192
	const indexCount = 64

	table := randomEmbeddingTable(vocab, hidden, 0xE142)
	indices := randomEmbeddingIndices(indexCount, vocab, 0xE143)
	output := make([]float32, indexCount*hidden)

	b.SetBytes(int64(indexCount * hidden * 4))
	b.ResetTimer()

	for b.Loop() {
		runLookupF32Generic(
			unsafe.Pointer(&table[0]),
			unsafe.Pointer(&indices[0]),
			unsafe.Pointer(&output[0]),
			vocab, hidden, indexCount,
		)
	}
}

func BenchmarkCopyRowF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const hidden = 8192

	table := randomEmbeddingTable(2, hidden, 0xE144)
	dst := make([]float32, hidden)

	b.SetBytes(int64(hidden * 4))
	b.ResetTimer()

	for b.Loop() {
		copyRowF32AVX512(&dst[0], &table[hidden], hidden)
	}
}
