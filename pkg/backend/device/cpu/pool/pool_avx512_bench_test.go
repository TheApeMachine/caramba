//go:build amd64

package pool

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkMaxPool2x2Stride2AVX512(b *testing.B) {
	if !cpu.X86.HasAVX512F {
		b.Skip("AVX-512F required")
	}

	config := DefaultPoolConfig()
	inHeight, inWidth := 64, 64
	outHeight := (inHeight - config.KernelH) / config.StrideH + 1
	outWidth := (inWidth - config.KernelW) / config.StrideW + 1
	input := make([]float32, inHeight*inWidth)
	output := make([]float32, outHeight*outWidth)

	b.SetBytes(int64(inHeight * inWidth * 4))
	b.ResetTimer()

	for b.Loop() {
		Pool2DFloat32Native(
			config, float32ViewFromSlice(input), float32ViewFromSlice(output),
			1, 1, inHeight, inWidth, outHeight, outWidth,
			true,
		)
	}
}

func BenchmarkMaxPool2x2Stride2Scalar(b *testing.B) {
	config := DefaultPoolConfig()
	inHeight, inWidth := 64, 64
	outHeight := (inHeight - config.KernelH) / config.StrideH + 1
	outWidth := (inWidth - config.KernelW) / config.StrideW + 1
	input := make([]float32, inHeight*inWidth)
	output := make([]float32, outHeight*outWidth)

	b.SetBytes(int64(inHeight * inWidth * 4))
	b.ResetTimer()

	for b.Loop() {
		Pool2DFloat32Scalar(
			config, input, output,
			1, 1, inHeight, inWidth, outHeight, outWidth,
			true,
		)
	}
}
