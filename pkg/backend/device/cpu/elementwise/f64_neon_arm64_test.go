//go:build arm64

package elementwise

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestAddFloat64NEONParity(t *testing.T) {
	rng := rand.New(rand.NewSource(0xADD064))

	for _, count := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", count), func(t *testing.T) {
			left := make([]float64, count)
			right := make([]float64, count)
			got := make([]float64, count)
			want := make([]float64, count)

			for index := range left {
				left[index] = rng.NormFloat64()
				right[index] = rng.NormFloat64()
				want[index] = left[index] + right[index]
			}

			AddFloat64NEONAsm(&got[0], &left[0], &right[0], count)

			for index := range got {
				if got[index] != want[index] {
					t.Fatalf("lane %d got=%g want=%g", index, got[index], want[index])
				}
			}
		})
	}
}

func BenchmarkAddFloat64NEONAsm(b *testing.B) {
	for _, count := range []int{64, 1024, 8192, 65536} {
		count := count

		b.Run(fmt.Sprintf("N=%d", count), func(b *testing.B) {
			left := make([]float64, count)
			right := make([]float64, count)
			out := make([]float64, count)

			b.SetBytes(int64(count * 8 * 3))
			b.ResetTimer()

			for b.Loop() {
				AddFloat64NEONAsm(&out[0], &left[0], &right[0], count)
			}
		})
	}
}
