package reduction

import (
	"math"
	"math/rand"

	"github.com/theapemachine/caramba/pkg/dtype"
)

var parityNs = []int{1, 7, 64, 1024, 8192}

func randomFloat32Slice(count int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, count)

	for index := range out {
		out[index] = float32((rng.Float64() - 0.5) * math.Pow(10, rng.Float64()*4-2))
	}

	return out
}

func randomBF16Slice(count int, seed int64) []dtype.BF16 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]dtype.BF16, count)

	for index := range out {
		bits := uint32(rng.Uint32())
		bits = bits & 0x7FFFFFFF
		bits = (bits & 0x807FFFFF) | (uint32(0x3E+rng.Intn(5)) << 23)

		if rng.Intn(2) == 0 {
			bits |= 0x80000000
		}

		out[index] = dtype.BF16(bits >> 16)
	}

	return out
}

func randomF16Slice(count int, seed int64) []dtype.F16 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]dtype.F16, count)

	for index := range out {
		bits := uint16(rng.Uint32())
		bits = (bits & 0x83FF) | (uint16(0x0E+rng.Intn(5)) << 10)
		out[index] = dtype.F16(bits)
	}

	return out
}
