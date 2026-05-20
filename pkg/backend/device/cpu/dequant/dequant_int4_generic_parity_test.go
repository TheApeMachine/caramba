package dequant

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func referenceDequantInt4(dst []float32, pairs tensor.Int4Vector, scale float32, zeroPoint int8) {
	for index := range dst {
		nibble := pairs.Get(index)
		dst[index] = float32(int(nibble)-int(zeroPoint)) * scale
	}
}

func int4BytesFromLength(length int, seed int64) []byte {
	byteCount := (length + 1) / 2
	bytes := make([]byte, byteCount)
	rng := rand.New(rand.NewSource(seed))

	for index := range bytes {
		bytes[index] = byte(rng.Uint32())
	}

	return bytes
}

func int4VectorFromBytes(bytes []byte, length int) tensor.Int4Vector {
	pairs := make([]dtype.Int4Pair, len(bytes))

	for index, value := range bytes {
		pairs[index] = dtype.Int4Pair(value)
	}

	return tensor.NewInt4Vector(pairs, length)
}

func TestDequantInt4GenericParity(t *testing.T) {
	convey.Convey("Given dequantInt4Generic", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match the reference dequant for N=%d", length), func() {
				bytes := int4BytesFromLength(length, 0x401e+int64(length))
				pairs := int4VectorFromBytes(bytes, length)

				const scale = float32(0.0625)
				const zeroPoint = int8(3)

				want := make([]float32, length)
				got := make([]float32, length)

				referenceDequantInt4(want, pairs, scale, zeroPoint)
				dequantInt4Generic(got, pairs, scale, zeroPoint)

				for index := range want {
					if want[index] != got[index] {
						t.Fatalf(
							"N=%d lane %d want=%g got=%g nibble=%d",
							length, index, want[index], got[index], pairs.Get(index),
						)
					}
				}
			})
		}
	})
}

func BenchmarkDequantInt4Generic(b *testing.B) {
	const length = 8192

	bytes := int4BytesFromLength(length, 1)
	pairs := int4VectorFromBytes(bytes, length)
	destination := make([]float32, length)

	b.SetBytes(int64(length / 2))
	b.ResetTimer()

	for b.Loop() {
		dequantInt4Generic(destination, pairs, 0.0625, 3)
	}
}
