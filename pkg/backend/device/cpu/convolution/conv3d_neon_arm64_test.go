//go:build arm64

package convolution

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

func TestConv3DFloat32NEONParitySizes(t *testing.T) {
	config := DefaultConv3DConfig()
	cases := []struct {
		inD, inH, inW, kD, kH, kW int
	}{
		{1, 7, 7, 1, 3, 3},
		{4, 8, 8, 3, 3, 3},
		{7, 16, 16, 3, 5, 5},
	}

	for _, testCase := range cases {
		label := fmt.Sprintf("d=%d_h=%d_k=%dx%dx%d", testCase.inD, testCase.inH, testCase.kD, testCase.kH, testCase.kW)
		t.Run(label, func(t *testing.T) {
			batch, inChannels := 1, 2
			outChannels := 2
			outD := testCase.inD - testCase.kD + 1
			outH := testCase.inH - testCase.kH + 1
			outW := testCase.inW - testCase.kW + 1
			input := randFloat32Slice(batch*inChannels*testCase.inD*testCase.inH*testCase.inW, 0x3E0)
			weight := randFloat32Slice(outChannels*inChannels*testCase.kD*testCase.kH*testCase.kW, 0x3E1)
			bias := randFloat32Slice(outChannels, 0x3E2)
			got := make([]float32, batch*outChannels*outD*outH*outW)
			want := make([]float32, len(got))

			Conv3DFloat32Native(
				config,
				convFloat32Pointer(input), convFloat32Pointer(weight),
				convFloat32Pointer(bias), convFloat32Pointer(got),
				batch, inChannels, testCase.inD, testCase.inH, testCase.inW,
				outChannels, testCase.kD, testCase.kH, testCase.kW, outD, outH, outW,
			)
			Conv3DFloat32Scalar(
				config,
				convFloat32Pointer(input), convFloat32Pointer(weight),
				convFloat32Pointer(bias), convFloat32Pointer(want),
				batch, inChannels, testCase.inD, testCase.inH, testCase.inW,
				outChannels, testCase.kD, testCase.kH, testCase.kW, outD, outH, outW,
			)

			assertFloat32SlicesNear(t, got, want, 1e-4)
		})
	}
}

func BenchmarkConv3DFloat32Native(b *testing.B) {
	config := DefaultConv3DConfig()
	batch, inChannels, inD, inH, inW := 1, 4, 8, 16, 16
	outChannels, kD, kH, kW := 4, 3, 3, 3
	outD, outH, outW := inD-kD+1, inH-kH+1, inW-kW+1
	input := randFloat32Slice(batch*inChannels*inD*inH*inW, 1)
	weight := randFloat32Slice(outChannels*inChannels*kD*kH*kW, 2)
	bias := randFloat32Slice(outChannels, 3)
	output := make([]float32, batch*outChannels*outD*outH*outW)

	b.ResetTimer()

	for b.Loop() {
		Conv3DFloat32Native(
			config,
			convFloat32Pointer(input), convFloat32Pointer(weight),
			convFloat32Pointer(bias), convFloat32Pointer(output),
			batch, inChannels, inD, inH, inW,
			outChannels, kD, kH, kW, outD, outH, outW,
		)
	}
}

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
