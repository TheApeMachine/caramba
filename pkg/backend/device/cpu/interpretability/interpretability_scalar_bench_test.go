package interpretability

import "testing"

func BenchmarkActivationSteerFloat32Scalar(b *testing.B) {
	const benchLen = 8192
	const coefficient = float32(0.25)

	base, direction := randomSteerVectors(benchLen, 0x2A30)
	destination := make([]float32, benchLen)

	b.SetBytes(int64(benchLen * 4))
	b.ResetTimer()

	for b.Loop() {
		ActivationSteerFloat32Scalar(destination, base, direction, coefficient)
	}
}
