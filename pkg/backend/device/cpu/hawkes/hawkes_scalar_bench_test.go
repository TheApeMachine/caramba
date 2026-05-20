//go:build arm64

package hawkes

import "testing"

func BenchmarkHawkesIntensityScalar(b *testing.B) {
	eventTimes := hawkesEventTimesForTest(8192, 0x2700)
	queryTimes := hawkesSingleQueryAfterEvents(eventTimes, 0x2701)
	output := make([]float32, 1)

	b.ResetTimer()

	for b.Loop() {
		HawkesIntensityScalar(eventTimes, queryTimes, output, 0.1, 0.5, 1.0)
	}
}

func BenchmarkHawkesIntensityNative(b *testing.B) {
	eventTimes := hawkesEventTimesForTest(8192, 0x2702)
	queryTimes := hawkesSingleQueryAfterEvents(eventTimes, 0x2703)
	output := make([]float32, 1)

	b.ResetTimer()

	for b.Loop() {
		HawkesIntensityNative(eventTimes, queryTimes, output, 0.1, 0.5, 1.0)
	}
}

func BenchmarkHawkesLogLikelihoodScalar(b *testing.B) {
	eventTimes := hawkesEventTimesForTest(8192, 0x2704)
	output := make([]float32, 1)

	b.ResetTimer()

	for b.Loop() {
		HawkesLogLikelihoodScalar(eventTimes, 10.0, 0.2, 0.5, 1.0, output)
	}
}

func BenchmarkHawkesLogLikelihoodNative(b *testing.B) {
	eventTimes := hawkesEventTimesForTest(8192, 0x2705)
	output := make([]float32, 1)

	b.ResetTimer()

	for b.Loop() {
		HawkesLogLikelihoodNative(eventTimes, 10.0, 0.2, 0.5, 1.0, output)
	}
}

func BenchmarkHawkesKernelMatrixScalar(b *testing.B) {
	eventCount := 128
	eventTimes := hawkesEventTimesForTest(eventCount, 0x2706)
	output := make([]float32, eventCount*eventCount)

	b.ResetTimer()

	for b.Loop() {
		HawkesKernelMatrixScalar(eventTimes, output, 0.5, 1.0)
	}
}

func BenchmarkHawkesKernelMatrixNative(b *testing.B) {
	eventCount := 128
	eventTimes := hawkesEventTimesForTest(eventCount, 0x2707)
	output := make([]float32, eventCount*eventCount)

	b.ResetTimer()

	for b.Loop() {
		HawkesKernelMatrixNative(eventTimes, output, 0.5, 1.0)
	}
}
