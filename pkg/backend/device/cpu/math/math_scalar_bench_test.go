package math

import "testing"

func BenchmarkLogSumExpRowGeneric(b *testing.B) {
	row := randomMathFloat32(8192, 0x2230)

	b.ResetTimer()

	for b.Loop() {
		_ = LogSumExpRowGeneric(row)
	}
}

func BenchmarkOuterGeneric(b *testing.B) {
	left := randomMathFloat32(64, 0x2231)
	right := randomMathFloat32(128, 0x2232)
	output := make([]float32, len(left)*len(right))

	b.ResetTimer()

	for b.Loop() {
		OuterGeneric(left, right, output)
	}
}
