//go:build amd64

package pool

import (
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512PoolAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestMaxPool2x2Stride2AVX512Parity(t *testing.T) {
	if !avx512PoolAvailable() {
		t.Skip("AVX-512F required")
	}

	config := DefaultPoolConfig()
	inHeight, inWidth := 16, 16
	outHeight := (inHeight-config.KernelH)/config.StrideH + 1
	outWidth := (inWidth - config.KernelW) / config.StrideW + 1

	convey.Convey("Given MaxPool2x2Stride2 AVX-512", t, func() {
		for _, batchChannels := range []struct {
			batch, channels int
		}{
			{1, 1},
			{1, 3},
			{2, 2},
		} {
			convey.Convey("It should match scalar for batch/channel shape", func() {
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
					true,
				)
				Pool2DFloat32Scalar(
					config, input, want,
					batchChannels.batch, batchChannels.channels,
					inHeight, inWidth, outHeight, outWidth,
					true,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
			})
		}
	})
}

func TestAvgPool2x2Stride2AVX512Parity(t *testing.T) {
	if !avx512PoolAvailable() {
		t.Skip("AVX-512F required")
	}

	config := DefaultPoolConfig()
	inHeight, inWidth := 16, 16
	outHeight := (inHeight-config.KernelH)/config.StrideH + 1
	outWidth := (inWidth - config.KernelW) / config.StrideW + 1

	convey.Convey("Given AvgPool2x2Stride2 AVX-512", t, func() {
		input := make([]float32, inHeight*inWidth)
		for index := range input {
			input[index] = float32(index%53)*0.02 - 0.25
		}

		got := make([]float32, outHeight*outWidth)
		want := make([]float32, len(got))

		Pool2DFloat32Native(
			config, float32ViewFromSlice(input), float32ViewFromSlice(got),
			1, 1, inHeight, inWidth, outHeight, outWidth,
			false,
		)
		Pool2DFloat32Scalar(
			config, input, want,
			1, 1, inHeight, inWidth, outHeight, outWidth,
			false,
		)

		parity.AssertFloat32SlicesWithinULP(t, got, want, 2)
	})
}

func TestMaxPoolStride1AVX512Parity(t *testing.T) {
	if !avx512PoolAvailable() {
		t.Skip("AVX-512F required")
	}

	config := PoolConfig{
		KernelH: 3, KernelW: 3,
		StrideH: 1, StrideW: 1,
	}

	for _, length := range parity.Lengths {
		if length < 9 {
			continue
		}

		inSize := length
		inHeight := 1
		inWidth := inSize
		outHeight := inHeight - config.KernelH + 1
		outWidth := inWidth - config.KernelW + 1

		if outWidth < 1 {
			continue
		}

		input := make([]float32, inSize)
		for index := range input {
			input[index] = float32(index%31) * 0.03
		}

		got := make([]float32, outHeight*outWidth)
		want := make([]float32, len(got))

		Pool2DFloat32Native(
			config, float32ViewFromSlice(input), float32ViewFromSlice(got),
			1, 1, inHeight, inWidth, outHeight, outWidth,
			true,
		)
		Pool2DFloat32Scalar(
			config, input, want,
			1, 1, inHeight, inWidth, outHeight, outWidth,
			true,
		)

		parity.AssertFloat32SlicesWithinULP(t, got, want, 0)
	}
}

func float32ViewFromSlice(slice []float32) unsafe.Pointer {
	return unsafe.Pointer(unsafe.SliceData(slice))
}
