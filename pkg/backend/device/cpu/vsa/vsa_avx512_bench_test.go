//go:build amd64

package vsa

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkVsaBindF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const benchLen = 8192
	left, right := randomVSAVectors(benchLen, 0xCD01)
	output := make([]float32, benchLen)

	for b.Loop() {
		VsaBindF32AVX512(&output[0], &left[0], &right[0], benchLen)
	}
}

func BenchmarkVsaBundleF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const benchLen = 8192
	left, right := randomVSAVectors(benchLen, 0xCD02)
	output := make([]float32, benchLen)

	for b.Loop() {
		VsaBundleF32AVX512(&output[0], &left[0], &right[0], benchLen)
	}
}

func BenchmarkVsaPermuteCopyF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const benchLen = 8192
	input, _ := randomVSAVectors(benchLen, 0xCD03)
	output := make([]float32, benchLen)

	for b.Loop() {
		VsaPermuteCopyF32AVX512(&output[0], &input[0], benchLen)
	}
}

func BenchmarkVsaSimilarityF32AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	const benchLen = 8192
	left, right := randomVSAVectors(benchLen, 0xCD04)

	for b.Loop() {
		_ = VsaSimilarityF32AVX512(&left[0], &right[0], benchLen)
	}
}
