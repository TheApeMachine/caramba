//go:build arm64

package losses

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

/*
Pair-sum parity is not bitwise: parallel f32 accumulation in NEON
reorders additions relative to the scalar reference. Both paths use
f32 lanes with a final horizontal fold, so error grows with n and
input magnitude. The bound matches reduction/sum_f32_neon_arm64_test.
*/

func TestMseSumNEONAsmParity(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5E))

	for _, count := range []int{1, 7, 64, 1024, 8192} {
		t.Run(fmt.Sprintf("N=%d", count), func(t *testing.T) {
			predictions := make([]float32, count)
			targets := make([]float32, count)

			for index := range predictions {
				predictions[index] = float32(rng.NormFloat64())
				targets[index] = float32(rng.NormFloat64())
			}

			want := MseSumFloat32Scalar(predictions, targets)
			got := MseSumNEONAsm(&predictions[0], &targets[0], count)
			assertPairSumClose(t, count, want, got)
		})
	}
}

func TestMaeSumNEONAsmParity(t *testing.T) {
	rng := rand.New(rand.NewSource(0xAB))

	for _, count := range []int{1, 7, 64, 1024, 8192} {
		t.Run(fmt.Sprintf("N=%d", count), func(t *testing.T) {
			predictions := make([]float32, count)
			targets := make([]float32, count)

			for index := range predictions {
				predictions[index] = float32(rng.NormFloat64())
				targets[index] = float32(rng.NormFloat64())
			}

			want := MaeSumFloat32Scalar(predictions, targets)
			got := MaeSumNEONAsm(&predictions[0], &targets[0], count)
			assertPairSumClose(t, count, want, got)
		})
	}
}

func assertPairSumClose(t *testing.T, count int, want, got float32) {
	t.Helper()

	relative := math.Abs(float64(got-want)) / math.Max(math.Abs(float64(want)), 1.0)

	if relative > 1e-5 {
		t.Fatalf("N=%d want=%g got=%g diff=%g relative=%g",
			count, want, got, got-want, relative)
	}
}

func BenchmarkMseSumFloat32Native(b *testing.B) {
	count := 8192
	predictions := make([]float32, count)
	targets := make([]float32, count)

	for index := range predictions {
		predictions[index] = float32(index) * 0.01
		targets[index] = float32(index) * 0.02
	}

	b.SetBytes(int64(count * 8))
	b.ResetTimer()

	for b.Loop() {
		_ = MseSumFloat32Native(predictions, targets)
	}
}
