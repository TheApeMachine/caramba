//go:build amd64

package kernels

import (
	"testing"

	"golang.org/x/sys/cpu"
)

func TestConvTranspose2dTapAVX2AsmDirect(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 not available")
	}

	t.Run("asm_params_direct", func(t *testing.T) {
		input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		got := []float32{0, 0, 0, 0, 0, 0, 0, 0}

		convTranspose2dTapAVX2Asm(
			&got[0],
			2,
			&input[0],
			8,
		)

		assertFloat32SlicesNear(t, got, []float32{2, 4, 6, 8, 10, 12, 14, 16}, 0)
	})

	t.Run("single_tap", func(t *testing.T) {
		input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		weight := []float32{2}
		got := make([]float32, 8)
		want := []float32{2, 4, 6, 8, 10, 12, 14, 16}

		convTranspose2dStride1RowAVX2(
			got,
			input,
			weight,
			8,
			1, 1, 1, 16,
			0, 0,
		)

		assertFloat32SlicesNear(t, got, want, 0)
	})
}

func TestConvTranspose2DFloat32AVX2Parity(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 not available")
	}

	config := DefaultConv2DConfig()
	batch, inC, inH, inW := 1, 2, 8, 8
	outC, kH, kW := 2, 3, 3
	outH := (inH-1)*config.StrideH + kH
	outW := (inW-1)*config.StrideW + kW
	input := randFloat32Slice(batch*inC*inH*inW, 0x720)
	weight := randFloat32Slice(inC*outC*kH*kW, 0x721)
	bias := randFloat32Slice(outC, 0x722)
	got := make([]float32, batch*outC*outH*outW)
	want := make([]float32, len(got))

	convTranspose2DFloat32Native(
		config, input, weight, bias, got,
		batch, inC, inH, inW, outC, kH, kW, outH, outW,
	)
	convTranspose2DFloat32Scalar(
		config, input, weight, bias, want,
		batch, inC, inH, inW, outC, kH, kW, outH, outW,
	)

	assertFloat32SlicesNear(t, got, want, 1e-4)
}
