//go:build amd64

package hawkes

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func avx512HawkesBenchAvailable() bool {
	return cpu.X86.HasAVX512F
}

func BenchmarkHawkesExpSumFloat32AVX512Asm(b *testing.B) {
	if !avx512HawkesBenchAvailable() {
		b.Skip("AVX-512F required")
	}

	exponents := randomHawkesExponents(8192, 0x2500)

	b.ResetTimer()

	for b.Loop() {
		_ = HawkesExpSumFloat32AVX512Asm(&exponents[0], len(exponents))
	}
}

func BenchmarkHawkesScaledExpStoreFloat32AVX512Asm(b *testing.B) {
	if !avx512HawkesBenchAvailable() {
		b.Skip("AVX-512F required")
	}

	exponents := randomHawkesExponents(8192, 0x2501)
	output := make([]float32, 8192)
	alpha := float32(0.5)

	b.ResetTimer()

	for b.Loop() {
		HawkesScaledExpStoreFloat32AVX512Asm(&exponents[0], alpha, &output[0], len(output))
	}
}

func BenchmarkHawkesIntensityNative(b *testing.B) {
	eventTimes := hawkesEventTimesForTest(8192, 0x2502)
	queryTimes := hawkesSingleQueryAfterEvents(eventTimes, 0x2503)
	output := make([]float32, 1)

	b.ResetTimer()

	for b.Loop() {
		HawkesIntensityNative(eventTimes, queryTimes, output, 0.1, 0.5, 1.0)
	}
}

func BenchmarkHawkesIntensityScalar(b *testing.B) {
	eventTimes := hawkesEventTimesForTest(8192, 0x2504)
	queryTimes := hawkesSingleQueryAfterEvents(eventTimes, 0x2505)
	output := make([]float32, 1)

	b.ResetTimer()

	for b.Loop() {
		HawkesIntensityScalar(eventTimes, queryTimes, output, 0.1, 0.5, 1.0)
	}
}

func BenchmarkHawkesKernelMatrixNative(b *testing.B) {
	eventCount := 128
	eventTimes := hawkesEventTimesForTest(eventCount, 0x2506)
	output := make([]float32, eventCount*eventCount)

	b.ResetTimer()

	for b.Loop() {
		HawkesKernelMatrixNative(eventTimes, output, 0.5, 1.0)
	}
}

func BenchmarkHawkesLogLikelihoodNative(b *testing.B) {
	eventTimes := hawkesEventTimesForTest(8192, 0x2507)
	output := make([]float32, 1)

	b.ResetTimer()

	for b.Loop() {
		HawkesLogLikelihoodNative(eventTimes, 10.0, 0.2, 0.5, 1.0, output)
	}
}
