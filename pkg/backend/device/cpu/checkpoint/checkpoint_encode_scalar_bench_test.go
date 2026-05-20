package checkpoint

import "testing"

func BenchmarkEncodeFloat32DataScalar(b *testing.B) {
	const benchLen = 8192
	source := randomFloat32Vector(benchLen, 0x2930)
	destination := make([]uint8, benchLen*4)

	b.SetBytes(int64(benchLen * 4))
	b.ResetTimer()

	for b.Loop() {
		encodeFloat32DataScalar(destination, source)
	}
}
