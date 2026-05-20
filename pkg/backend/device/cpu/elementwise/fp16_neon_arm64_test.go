//go:build arm64

package elementwise

import (
	"fmt"
	"testing"

	"github.com/theapemachine/manifesto/dtype"
)

/*
FP16 elementwise parity. NEON widens via FCVTL/FCVTL2, computes in
f32, narrows via FCVTN/FCVTN2. Scalar reference uses dtype.Fromfloat32
and (*F16).Float32() which produce IEEE 754 round-to-nearest-even
results — same rounding mode as the hardware FCVTL/FCVTN. The two
paths should agree bit-for-bit at the f16 representation.
*/

func assertF16Equal(t *testing.T, op string, scalar, neon []dtype.F16) {
	t.Helper()

	for index := range scalar {
		if uint16(scalar[index]) == uint16(neon[index]) {
			continue
		}

		t.Fatalf("%s: lane %d scalar=0x%04x (%g) neon=0x%04x (%g)",
			op,
			index,
			uint16(scalar[index]), scalar[index].Float32(),
			uint16(neon[index]), neon[index].Float32(),
		)
	}
}

func TestAddFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomF16Slice(n, 0x11+int64(n))
			right := randomF16Slice(n, 0x22+int64(n))

			scalar := make([]dtype.F16, n)
			for index := range scalar {
				scalar[index] = dtype.Fromfloat32(left[index].Float32() + right[index].Float32())
			}

			neon := make([]dtype.F16, n)
			AddFloat16Native(neon, left, right)

			assertF16Equal(t, "add", scalar, neon)
		})
	}
}

func TestSubFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomF16Slice(n, 0x33+int64(n))
			right := randomF16Slice(n, 0x44+int64(n))

			scalar := make([]dtype.F16, n)
			for index := range scalar {
				scalar[index] = dtype.Fromfloat32(left[index].Float32() - right[index].Float32())
			}

			neon := make([]dtype.F16, n)
			SubFloat16Native(neon, left, right)

			assertF16Equal(t, "sub", scalar, neon)
		})
	}
}

func TestMulFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomF16Slice(n, 0x55+int64(n))
			right := randomF16Slice(n, 0x66+int64(n))

			scalar := make([]dtype.F16, n)
			for index := range scalar {
				scalar[index] = dtype.Fromfloat32(left[index].Float32() * right[index].Float32())
			}

			neon := make([]dtype.F16, n)
			MulFloat16Native(neon, left, right)

			assertF16Equal(t, "mul", scalar, neon)
		})
	}
}

func TestDivFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomF16Slice(n, 0x77+int64(n))
			right := randomF16Slice(n, 0x88+int64(n))

			scalar := make([]dtype.F16, n)
			for index := range scalar {
				scalar[index] = dtype.Fromfloat32(left[index].Float32() / right[index].Float32())
			}

			neon := make([]dtype.F16, n)
			DivFloat16Native(neon, left, right)

			assertF16Equal(t, "div", scalar, neon)
		})
	}
}

func TestMaxFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomF16Slice(n, 0x99+int64(n))
			right := randomF16Slice(n, 0xAA+int64(n))

			scalar := make([]dtype.F16, n)
			for index := range scalar {
				l := left[index].Float32()
				r := right[index].Float32()
				scalar[index] = dtype.Fromfloat32(r)

				if l > r {
					scalar[index] = dtype.Fromfloat32(l)
				}
			}

			neon := make([]dtype.F16, n)
			MaxFloat16Native(neon, left, right)

			assertF16Equal(t, "max", scalar, neon)
		})
	}
}

func TestMinFloat16NEONAsmParity(t *testing.T) {
	for _, n := range elementwiseParityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomF16Slice(n, 0xBB+int64(n))
			right := randomF16Slice(n, 0xCC+int64(n))

			scalar := make([]dtype.F16, n)
			for index := range scalar {
				l := left[index].Float32()
				r := right[index].Float32()
				scalar[index] = dtype.Fromfloat32(r)

				if l < r {
					scalar[index] = dtype.Fromfloat32(l)
				}
			}

			neon := make([]dtype.F16, n)
			MinFloat16Native(neon, left, right)

			assertF16Equal(t, "min", scalar, neon)
		})
	}
}

func BenchmarkAddFloat16NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			left := randomF16Slice(n, 1)
			right := randomF16Slice(n, 2)
			out := make([]dtype.F16, n)

			b.SetBytes(int64(n * 2 * 3))
			b.ResetTimer()

			for b.Loop() {
				AddFloat16Native(out, left, right)
			}
		})
	}
}

func BenchmarkMulFloat16NEONAsm(b *testing.B) {
	for _, n := range []int{64, 1024, 8192, 65536} {
		n := n

		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			left := randomF16Slice(n, 1)
			right := randomF16Slice(n, 2)
			out := make([]dtype.F16, n)

			b.SetBytes(int64(n * 2 * 3))
			b.ResetTimer()

			for b.Loop() {
				MulFloat16Native(out, left, right)
			}
		})
	}
}
