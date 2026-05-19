//go:build amd64 || 386

package activation

import (
	"math"
	"testing"
	"math/rand"
	"golang.org/x/sys/cpu"
)

func assertSoftmaxParity(t *testing.T, count int) {
	if !cpu.X86.HasSSE2 {
		t.Skip("SSE2 not supported")
	}
	src := make([]float32, count)
	dstSSE := make([]float32, count)
	dstGen := make([]float32, count)

	for i := 0; i < count; i++ {
		src[i] = rand.Float32()*2 - 1
	}

	SoftmaxF32SSE2(&dstSSE[0], &src[0], count)
	SoftmaxF32Generic(&dstGen[0], &src[0], count)

	for i := 0; i < count; i++ {
		if math.Abs(float64(dstSSE[i]-dstGen[i])) > 1e-5 {
			t.Fatalf("mismatch at index %d: sse2=%f generic=%f", i, dstSSE[i], dstGen[i])
		}
	}
}

func TestSoftmaxF32SSE2(t *testing.T) {
	sizes := []int{1, 7, 64, 1024, 8192}
	for _, n := range sizes {
		assertSoftmaxParity(t, n)
	}
}

func BenchmarkSoftmaxF32SSE2(b *testing.B) {
	if !cpu.X86.HasSSE2 {
		b.Skip("SSE2 not supported")
	}
	count := 1024
	src := make([]float32, count)
	dst := make([]float32, count)
	for i := 0; i < count; i++ {
		src[i] = rand.Float32()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SoftmaxF32SSE2(&dst[0], &src[0], count)
	}
}

func BenchmarkSoftmaxF32Generic(b *testing.B) {
	count := 1024
	src := make([]float32, count)
	dst := make([]float32, count)
	for i := 0; i < count; i++ {
		src[i] = rand.Float32()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SoftmaxF32Generic(&dst[0], &src[0], count)
	}
}
