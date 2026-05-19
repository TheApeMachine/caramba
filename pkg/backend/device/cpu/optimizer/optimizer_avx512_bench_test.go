//go:build amd64

package optimizer

import (
	"testing"
)

func BenchmarkAdamStepSlicesAVX512(b *testing.B) {
	length := 8192
	config := DefaultAdamConfig()
	params := randFloat32Slice(length, 0x2400)
	grad := randFloat32Slice(length, 0x2401)
	first := randFloat32Slice(length, 0x2402)
	second := randFloat32Slice(length, 0x2403)
	out := make([]float32, length)

	b.ResetTimer()

	for b.Loop() {
		adamStepSlicesAVX512(config, params, grad, first, second, out)
	}
}

func BenchmarkSgdStepSlicesAVX512(b *testing.B) {
	length := 8192
	config := DefaultSGDConfig()
	params := randFloat32Slice(length, 0x2410)
	grad := randFloat32Slice(length, 0x2411)
	momentum := randFloat32Slice(length, 0x2412)
	out := make([]float32, length)

	b.ResetTimer()

	for b.Loop() {
		sgdStepSlicesAVX512(config, params, grad, momentum, out)
	}
}

func BenchmarkAdamStepFloat32AVX512Asm(b *testing.B) {
	length := 8192
	config := DefaultAdamConfig()
	params := randFloat32Slice(length, 0x2420)
	grad := randFloat32Slice(length, 0x2421)
	first := randFloat32Slice(length, 0x2422)
	second := randFloat32Slice(length, 0x2423)
	out := make([]float32, length)
	beta1Correction := float32(0.9)
	beta2Correction := float32(0.999)

	b.ResetTimer()

	for b.Loop() {
		AdamStepFloat32AVX512Asm(
			&params[0], &grad[0], &first[0], &second[0], &out[0],
			length,
			config.LearningRate, config.Beta1, config.Beta2, config.Epsilon,
			beta1Correction, beta2Correction,
		)
	}
}
