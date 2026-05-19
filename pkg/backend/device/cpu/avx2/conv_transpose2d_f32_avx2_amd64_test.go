//go:build amd64

package avx2

import (
	"math"
	"math/rand"
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

	ConvTranspose2DFloat32(
		config, input, weight, bias, got,
		batch, inC, inH, inW, outC, kH, kW, outH, outW,
	)
	ConvTranspose2DFloat32Scalar(
		config, input, weight, bias, want,
		batch, inC, inH, inW, outC, kH, kW, outH, outW,
	)

	assertFloat32SlicesNear(t, got, want, 1e-4)
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
	tolerance float64,
) {
	testing.Helper()

	if len(got) != len(want) {
		testing.Fatalf("length mismatch got=%d want=%d", len(got), len(want))
	}

	for index := range got {
		diff := math.Abs(float64(got[index] - want[index]))

		if diff > tolerance {
			testing.Fatalf(
				"lane %d got=%g want=%g diff=%g",
				index, got[index], want[index], diff,
			)
		}
	}
}
