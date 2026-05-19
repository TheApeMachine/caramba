//go:build amd64

package causal

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func requireAVX512CausalBench(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}
}

func BenchmarkCateF32AVX512(b *testing.B) {
	requireAVX512CausalBench(b)

	const length = 8192
	treated := randomCausalFloat32Slice(length, 1)
	control := randomCausalFloat32Slice(length, 2)
	out := make([]float32, length)

	for b.Loop() {
		CateFloat32AVX512Asm(&treated[0], &control[0], &out[0], length)
	}
}

func BenchmarkCounterfactualF32AVX512(b *testing.B) {
	requireAVX512CausalBench(b)

	const length = 8192
	observedY := randomCausalFloat32Slice(length, 1)
	observedX := randomCausalFloat32Slice(length, 2)
	counterfactualX := randomCausalFloat32Slice(length, 3)
	out := make([]float32, length)

	for b.Loop() {
		CounterfactualFloat32AVX512Asm(
			&out[0], &observedY[0], &observedX[0], &counterfactualX[0],
			0.5, length,
		)
	}
}

func BenchmarkStridedDotF32AVX512(b *testing.B) {
	requireAVX512CausalBench(b)

	const length = 8192
	const stride = 7
	values := randomCausalFloat32Slice(length*stride, 1)
	weights := randomCausalFloat32Slice(length, 2)

	for b.Loop() {
		_ = StridedDotFloat32AVX512Asm(&values[0], stride, &weights[0], length)
	}
}
