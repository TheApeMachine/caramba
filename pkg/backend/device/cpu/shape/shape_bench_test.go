package shape

import "testing"

func BenchmarkCopyContiguousFloat32NativeHost(b *testing.B) {
	length := 8192
	source := randomShapeFloat32(length, 0x1730)
	destination := make([]float32, length)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		CopyContiguousFloat32Native(destination, source)
	}
}

func BenchmarkWhereFloat32NativeHost(b *testing.B) {
	length := 8192
	positive := randomShapeFloat32(length, 0x1731)
	negative := randomShapeFloat32(length, 0x1732)
	mask := shapeMaskBytes(length, 0x1733)
	destination := make([]float32, length)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		WhereFloat32Native(destination, positive, negative, mask)
	}
}

func BenchmarkMaskedFillFloat32NativeHost(b *testing.B) {
	length := 8192
	input := randomShapeFloat32(length, 0x1734)
	mask := shapeMaskBytes(length, 0x1735)
	destination := make([]float32, length)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		MaskedFillFloat32Native(destination, input, 1.5, mask)
	}
}
