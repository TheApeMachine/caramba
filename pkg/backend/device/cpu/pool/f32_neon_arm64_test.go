//go:build arm64

package pool

import (
	"testing"
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func TestAvgPool2x2Stride2NEONParity(t *testing.T) {
	config := DefaultPoolConfig()
	inHeight, inWidth := 16, 16
	outHeight := (inHeight - config.KernelH) / config.StrideH + 1
	outWidth := (inWidth - config.KernelW) / config.StrideW + 1

	for _, batchChannels := range []struct {
		batch, channels int
	}{
		{1, 1},
		{1, 3},
		{2, 2},
	} {
		input := make([]float32, batchChannels.batch*batchChannels.channels*inHeight*inWidth)
		for index := range input {
			input[index] = float32(index%97)*0.01 - 0.5
		}

		got := make([]float32, batchChannels.batch*batchChannels.channels*outHeight*outWidth)
		want := make([]float32, len(got))

		Pool2DFloat32Native(
			config, float32ViewFromSlice(input), float32ViewFromSlice(got),
			batchChannels.batch, batchChannels.channels,
			inHeight, inWidth, outHeight, outWidth,
			false,
		)
		Pool2DFloat32Scalar(
			config, input, want,
			batchChannels.batch, batchChannels.channels,
			inHeight, inWidth, outHeight, outWidth,
			false,
		)

		parity.AssertFloat32SlicesWithinULP(t, got, want, 2)
	}
}

func BenchmarkAvgPool2x2Stride2NEON(b *testing.B) {
	config := DefaultPoolConfig()
	inHeight, inWidth := 64, 64
	outHeight := (inHeight - config.KernelH) / config.StrideH + 1
	outWidth := (inWidth - config.KernelW) / config.StrideW + 1
	input := make([]float32, inHeight*inWidth)
	output := make([]float32, outHeight*outWidth)

	for b.Loop() {
		Pool2DFloat32Native(
			config, float32ViewFromSlice(input), float32ViewFromSlice(output),
			1, 1, inHeight, inWidth, outHeight, outWidth,
			false,
		)
	}
}

func float32ViewFromSlice(slice []float32) unsafe.Pointer {
	return unsafe.Pointer(unsafe.SliceData(slice))
}
