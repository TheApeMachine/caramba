//go:build arm64

package kernels

import (
	"fmt"
	"math/rand"
	"testing"
)

/*
INT8 dot product parity. NEON path uses SDOT (FEAT_DotProd, ARMv8.2+):
each instruction computes 4 partial int8×int8 dots into 4 int32 lanes,
accumulating into the destination register. Scalar reference uses
sum(int32(a[i]) * int32(b[i])). Both produce identical int32 results
because the math is exact integer arithmetic.

Note: if the host doesn't have FEAT_DotProd, this will SIGILL at
runtime. Most arm64 servers (Graviton 3+, Apple M-series, modern
phones) support it.
*/

func TestDotInt8NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			rng := rand.New(rand.NewSource(0x180D + int64(n)))
			a := make([]int8, n)
			b := make([]int8, n)
			for index := range a {
				a[index] = int8(rng.Intn(256) - 128)
				b[index] = int8(rng.Intn(256) - 128)
			}

			var scalar int32
			for index := range a {
				scalar += int32(a[index]) * int32(b[index])
			}

			neon := dotInt8Native(a, b)

			if scalar != neon {
				t.Fatalf("N=%d scalar=%d neon=%d", n, scalar, neon)
			}
		})
	}
}

func BenchmarkDotInt8NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			a := make([]int8, n)
			c := make([]int8, n)
			rng := rand.New(rand.NewSource(1))
			for index := range a {
				a[index] = int8(rng.Intn(256) - 128)
				c[index] = int8(rng.Intn(256) - 128)
			}

			b.SetBytes(int64(n * 2))
			b.ResetTimer()

			for b.Loop() {
				_ = dotInt8Native(a, c)
			}
		})
	}
}

func BenchmarkDotInt8Scalar(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			a := make([]int8, n)
			c := make([]int8, n)
			rng := rand.New(rand.NewSource(1))
			for index := range a {
				a[index] = int8(rng.Intn(256) - 128)
				c[index] = int8(rng.Intn(256) - 128)
			}

			b.SetBytes(int64(n * 2))
			b.ResetTimer()

			for b.Loop() {
				var sum int32
				for index := range a {
					sum += int32(a[index]) * int32(c[index])
				}
				_ = sum
			}
		})
	}
}
