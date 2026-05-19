package kernels

import (
	"math/rand"

	"github.com/theapemachine/caramba/pkg/dtype"
)

func randomBF16Slice(n int, seed int64) []dtype.BF16 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]dtype.BF16, n)

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

func randomF16Slice(n int, seed int64) []dtype.F16 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]dtype.F16, n)

	for index := range out {
		bits := uint16(rng.Uint32())
		bits = (bits & 0x83FF) | (uint16(0x0E+rng.Intn(5)) << 10)
		out[index] = dtype.F16(bits)
	}

	return out
}
