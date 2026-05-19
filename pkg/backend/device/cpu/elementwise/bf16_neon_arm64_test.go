//go:build arm64

package elementwise

import (
	"fmt"
	"testing"

	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
BF16 elementwise parity. The NEON path is widen-via-bit-shuffle,
compute-in-f32, narrow-via-bit-shuffle. The scalar reference does
exactly the same thing in Go. Both are bit-exact, so the test
asserts bitwise equality on the underlying uint16 representation
at every lane.
*/

func TestAddBFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomBF16Slice(n, 0xA110+int64(n))
			right := randomBF16Slice(n, 0xB220+int64(n))

			scalar := make([]dtype.BF16, n)

			for index := range scalar {
				sum := (&left[index]).Float32() + (&right[index]).Float32()
				scalar[index] = dtype.NewBfloat16FromFloat32(sum)
			}

			neon := make([]dtype.BF16, n)
			AddBFloat16Native(neon, left, right)

			assertBF16Equal(t, "add", scalar, neon)
		})
	}
}

func TestSubBFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomBF16Slice(n, 0xC330+int64(n))
			right := randomBF16Slice(n, 0xD440+int64(n))

			scalar := make([]dtype.BF16, n)

			for index := range scalar {
				diff := (&left[index]).Float32() - (&right[index]).Float32()
				scalar[index] = dtype.NewBfloat16FromFloat32(diff)
			}

			neon := make([]dtype.BF16, n)
			SubBFloat16Native(neon, left, right)

			assertBF16Equal(t, "sub", scalar, neon)
		})
	}
}

func TestMulBFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomBF16Slice(n, 0xE550+int64(n))
			right := randomBF16Slice(n, 0xF660+int64(n))

			scalar := make([]dtype.BF16, n)

			for index := range scalar {
				product := (&left[index]).Float32() * (&right[index]).Float32()
				scalar[index] = dtype.NewBfloat16FromFloat32(product)
			}

			neon := make([]dtype.BF16, n)
			MulBFloat16Native(neon, left, right)

			assertBF16Equal(t, "mul", scalar, neon)
		})
	}
}

func TestDivBFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomBF16Slice(n, 0x1770+int64(n))
			right := randomBF16Slice(n, 0x2880+int64(n))

			scalar := make([]dtype.BF16, n)

			for index := range scalar {
				quotient := (&left[index]).Float32() / (&right[index]).Float32()
				scalar[index] = dtype.NewBfloat16FromFloat32(quotient)
			}

			neon := make([]dtype.BF16, n)
			DivBFloat16Native(neon, left, right)

			assertBF16Equal(t, "div", scalar, neon)
		})
	}
}

func assertBF16Equal(t *testing.T, op string, scalar, neon []dtype.BF16) {
	t.Helper()

	for index := range scalar {
		if uint16(scalar[index]) == uint16(neon[index]) {
			continue
		}

		t.Fatalf("%s: lane %d scalar=0x%04x (%g) neon=0x%04x (%g)",
			op,
			index,
			uint16(scalar[index]), (&scalar[index]).Float32(),
			uint16(neon[index]), (&neon[index]).Float32(),
		)
	}
}

func TestMaxBFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomBF16Slice(n, 0x3990+int64(n))
			right := randomBF16Slice(n, 0x4AA0+int64(n))

			scalar := make([]dtype.BF16, n)

			for index := range scalar {
				leftValue := (&left[index]).Float32()
				rightValue := (&right[index]).Float32()
				scalar[index] = dtype.NewBfloat16FromFloat32(rightValue)

				if leftValue > rightValue {
					scalar[index] = dtype.NewBfloat16FromFloat32(leftValue)
				}
			}

			neon := make([]dtype.BF16, n)
			MaxBFloat16Native(neon, left, right)

			assertBF16Equal(t, "max", scalar, neon)
		})
	}
}

func TestMinBFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomBF16Slice(n, 0x5BB0+int64(n))
			right := randomBF16Slice(n, 0x6CC0+int64(n))

			scalar := make([]dtype.BF16, n)

			for index := range scalar {
				leftValue := (&left[index]).Float32()
				rightValue := (&right[index]).Float32()
				scalar[index] = dtype.NewBfloat16FromFloat32(rightValue)

				if leftValue < rightValue {
					scalar[index] = dtype.NewBfloat16FromFloat32(leftValue)
				}
			}

			neon := make([]dtype.BF16, n)
			MinBFloat16Native(neon, left, right)

			assertBF16Equal(t, "min", scalar, neon)
		})
	}
}

func BenchmarkAddBFloat16NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			left := randomBF16Slice(n, 1)
			right := randomBF16Slice(n, 2)
			out := make([]dtype.BF16, n)

			b.SetBytes(int64(n * 2 * 3))
			b.ResetTimer()

			for b.Loop() {
				AddBFloat16Native(out, left, right)
			}
		})
	}
}

func BenchmarkMulBFloat16NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			left := randomBF16Slice(n, 1)
			right := randomBF16Slice(n, 2)
			out := make([]dtype.BF16, n)

			b.SetBytes(int64(n * 2 * 3))
			b.ResetTimer()

			for b.Loop() {
				MulBFloat16Native(out, left, right)
			}
		})
	}
}
