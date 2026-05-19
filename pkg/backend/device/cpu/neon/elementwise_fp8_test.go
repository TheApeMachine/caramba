package neon

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
FP8 elementwise parity. Each kernel widens FP8→f32, runs the f32 NEON
op, and narrows back. The scalar reference does identical work
lane-by-lane. Both paths share the same widen and narrow functions, so
the test asserts bitwise equality on the underlying uint8.
*/

func randomF8E4M3Slice(n int, seed int64) []dtype.F8E4M3 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]dtype.F8E4M3, n)

	for index := range out {
		// Avoid the canonical NaN encoding (0x7F / 0xFF) so binary
		// ops stay in well-defined territory for parity.
		for {
			byteValue := uint8(rng.Uint32())

			if byteValue&0x7F != 0x7F {
				out[index] = dtype.F8E4M3(byteValue)
				break
			}
		}
	}

	return out
}

func randomF8E5M2Slice(n int, seed int64) []dtype.F8E5M2 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]dtype.F8E5M2, n)

	for index := range out {
		for {
			byteValue := uint8(rng.Uint32())
			// Avoid inf/NaN sentinels (exp = 0x1F).
			exponent := (byteValue >> 2) & 0x1F

			if exponent != 0x1F {
				out[index] = dtype.F8E5M2(byteValue)
				break
			}
		}
	}

	return out
}

func TestAddF8E4M3Parity(t *testing.T) {
	for _, n := range parityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomF8E4M3Slice(n, 0x4001+int64(n))
			right := randomF8E4M3Slice(n, 0x4002+int64(n))

			scalar := make([]dtype.F8E4M3, n)
			for index := range scalar {
				scalar[index] = dtype.NewF8E4M3FromFloat32(left[index].Float32() + right[index].Float32())
			}

			leftTensor, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.Float8E4M3)
			rightTensor, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.Float8E4M3)
			outTensor, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.Float8E4M3)

			leftView, _ := leftTensor.Float8E4M3Native()
			rightView, _ := rightTensor.Float8E4M3Native()
			copy(leftView, left)
			copy(rightView, right)

			if err := runAddF8E4M3(leftTensor, rightTensor, outTensor); err != nil {
				t.Fatal(err)
			}

			outView, _ := outTensor.Float8E4M3Native()

			for index := range scalar {
				if uint8(scalar[index]) != uint8(outView[index]) {
					t.Fatalf("N=%d lane %d scalar=0x%02x (%g) kernel=0x%02x (%g)",
						n, index,
						uint8(scalar[index]), scalar[index].Float32(),
						uint8(outView[index]), outView[index].Float32(),
					)
				}
			}
		})
	}
}

func TestMulF8E4M3Parity(t *testing.T) {
	for _, n := range parityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomF8E4M3Slice(n, 0x5001+int64(n))
			right := randomF8E4M3Slice(n, 0x5002+int64(n))

			scalar := make([]dtype.F8E4M3, n)
			for index := range scalar {
				scalar[index] = dtype.NewF8E4M3FromFloat32(left[index].Float32() * right[index].Float32())
			}

			leftTensor, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.Float8E4M3)
			rightTensor, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.Float8E4M3)
			outTensor, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.Float8E4M3)

			leftView, _ := leftTensor.Float8E4M3Native()
			rightView, _ := rightTensor.Float8E4M3Native()
			copy(leftView, left)
			copy(rightView, right)

			if err := runMulF8E4M3(leftTensor, rightTensor, outTensor); err != nil {
				t.Fatal(err)
			}

			outView, _ := outTensor.Float8E4M3Native()

			for index := range scalar {
				if uint8(scalar[index]) != uint8(outView[index]) {
					t.Fatalf("N=%d lane %d scalar=0x%02x (%g) kernel=0x%02x (%g)",
						n, index,
						uint8(scalar[index]), scalar[index].Float32(),
						uint8(outView[index]), outView[index].Float32(),
					)
				}
			}
		})
	}
}

func TestAddF8E5M2Parity(t *testing.T) {
	for _, n := range parityNs {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			left := randomF8E5M2Slice(n, 0x6001+int64(n))
			right := randomF8E5M2Slice(n, 0x6002+int64(n))

			scalar := make([]dtype.F8E5M2, n)
			for index := range scalar {
				scalar[index] = dtype.NewF8E5M2FromFloat32(left[index].Float32() + right[index].Float32())
			}

			leftTensor, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.Float8E5M2)
			rightTensor, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.Float8E5M2)
			outTensor, _ := tensor.NewZeroed(mustShape([]int{n}), dtype.Float8E5M2)

			leftView, _ := leftTensor.Float8E5M2Native()
			rightView, _ := rightTensor.Float8E5M2Native()
			copy(leftView, left)
			copy(rightView, right)

			if err := runAddF8E5M2(leftTensor, rightTensor, outTensor); err != nil {
				t.Fatal(err)
			}

			outView, _ := outTensor.Float8E5M2Native()

			for index := range scalar {
				if uint8(scalar[index]) != uint8(outView[index]) {
					t.Fatalf("N=%d lane %d scalar=0x%02x (%g) kernel=0x%02x (%g)",
						n, index,
						uint8(scalar[index]), scalar[index].Float32(),
						uint8(outView[index]), outView[index].Float32(),
					)
				}
			}
		})
	}
}

func mustShape(dims []int) tensor.Shape {
	shape, _ := tensor.NewShape(dims)
	return shape
}
