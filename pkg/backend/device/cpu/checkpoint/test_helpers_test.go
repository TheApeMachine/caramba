package checkpoint

import (
	"math"
	"math/rand"
)

func randomFloat32Vector(count int, seed int64) []float32 {
	randomSource := rand.New(rand.NewSource(seed))
	values := make([]float32, count)

	for index := range values {
		values[index] = randomSource.Float32()*4 - 2
	}

	return values
}

func assertFloat32SliceEqual(
	testingTB interface {
		Helper()
		Fatalf(string, ...any)
	},
	got, want []float32,
) {
	testingTB.Helper()

	if len(got) != len(want) {
		testingTB.Fatalf("length mismatch got=%d want=%d", len(got), len(want))
	}

	for index := range got {
		if got[index] != want[index] {
			testingTB.Fatalf(
				"index %d got=%g want=%g",
				index, got[index], want[index],
			)
		}
	}
}

func assertUint8PayloadEqual(
	testingTB interface {
		Helper()
		Fatalf(string, ...any)
	},
	got, want []uint8,
) {
	testingTB.Helper()

	if len(got) != len(want) {
		testingTB.Fatalf("length mismatch got=%d want=%d", len(got), len(want))
	}

	for index := range got {
		if got[index] != want[index] {
			testingTB.Fatalf("index %d got=%d want=%d", index, got[index], want[index])
		}
	}
}

func float32SliceFromPayload(payload []uint8) []float32 {
	values := make([]float32, len(payload)/4)

	for index := range values {
		bits := uint32(payload[index*4]) |
			uint32(payload[index*4+1])<<8 |
			uint32(payload[index*4+2])<<16 |
			uint32(payload[index*4+3])<<24
		values[index] = math.Float32frombits(bits)
	}

	return values
}
