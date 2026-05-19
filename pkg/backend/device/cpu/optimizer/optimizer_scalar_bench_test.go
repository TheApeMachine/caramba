package optimizer

import "testing"

func BenchmarkAdamStepSlicesScalar(b *testing.B) {
	length := 8192
	config := DefaultAdamConfig()
	params := randFloat32Slice(length, 0x2800)
	grad := randFloat32Slice(length, 0x2801)
	first := randFloat32Slice(length, 0x2802)
	second := randFloat32Slice(length, 0x2803)
	out := make([]float32, length)

	b.ResetTimer()

	for b.Loop() {
		adamStepSlicesScalar(config, params, grad, first, second, out)
	}
}
