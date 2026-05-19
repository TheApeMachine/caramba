package masking

import (
	"testing"
	"unsafe"
)

func BenchmarkApplyMaskFloat32Native(b *testing.B) {
	length := 8192
	input := randomMaskingFloat32(length, 0x1850)
	mask := randomMaskingFloat32(length, 0x1851)
	output := make([]float32, length)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		ApplyMaskFloat32Native(
			unsafe.Pointer(&input[0]),
			unsafe.Pointer(&mask[0]),
			unsafe.Pointer(&output[0]),
			length,
		)
	}
}

func BenchmarkCausalMaskFloat32Native(b *testing.B) {
	side := 128
	output := make([]float32, side*side)

	b.SetBytes(int64(side * side * 4))
	b.ResetTimer()

	for b.Loop() {
		CausalMaskFloat32Native(unsafe.Pointer(&output[0]), side, side)
	}
}

func BenchmarkALiBiBiasFloat32Native(b *testing.B) {
	side := 128
	scores := randomMaskingScores(side, side, 0x1852)
	slope := []float32{0.125}
	output := make([]float32, side*side)

	b.SetBytes(int64(side * side * 4))
	b.ResetTimer()

	for b.Loop() {
		ALiBiBiasFloat32Native(
			unsafe.Pointer(&scores[0]),
			unsafe.Pointer(&slope[0]),
			unsafe.Pointer(&output[0]),
			side,
			side,
		)
	}
}
