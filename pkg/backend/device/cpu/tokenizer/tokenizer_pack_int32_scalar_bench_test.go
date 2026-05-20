package tokenizer

import "testing"

func BenchmarkPackInt32Scalar(b *testing.B) {
	const benchLen = 8192
	source := randomInt32Vector(benchLen, 0x2810)
	destination := make([]int32, benchLen)

	b.SetBytes(int64(benchLen * 4))
	b.ResetTimer()

	for b.Loop() {
		PackInt32Scalar(destination, source)
	}
}
