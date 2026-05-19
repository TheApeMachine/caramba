//go:build arm64

package neon

import (
	"math/rand"
	"testing"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func convFloat32Pointer(slice []float32) unsafe.Pointer {
	return unsafe.Pointer(unsafe.SliceData(slice))
}

func randFloat32Slice(elementCount int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	slice := make([]float32, elementCount)

	for index := range slice {
		slice[index] = float32(rng.NormFloat64()) * 0.1
	}

	return slice
}

func assertFloat32SlicesNear(
	testing *testing.T,
	got, want []float32,
	maxULP int,
) {
	testing.Helper()

	parity.AssertFloat32SlicesWithinULP(testing, got, want, maxULP)
}
