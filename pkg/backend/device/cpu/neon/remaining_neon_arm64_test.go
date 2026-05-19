//go:build arm64

package neon

import (
	"fmt"
	"math"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/attention"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/convolution"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/dropout"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/hawkes"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/matmul"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/pool"
)

var paritySizes = []int{1, 7, 64, 1024, 8192}

const maxULPElementary = 2

const maxULPReduction = 4

func TestConv2DFloat32NEONParity(t *testing.T) {
	config := convolution.DefaultConv2DConfig()
	cases := []struct {
		batch, inC, inH, inW, outC, kH, kW int
	}{
		{1, 1, 8, 8, 1, 3, 3},
		{1, 3, 16, 16, 2, 3, 3},
		{2, 4, 32, 32, 4, 5, 5},
	}

	for _, testCase := range cases {
		label := fmt.Sprintf("b=%d_c=%d_h=%d", testCase.batch, testCase.inC, testCase.inH)
		t.Run(label, func(t *testing.T) {
			outH := testCase.inH - testCase.kH + 1
			outW := testCase.inW - testCase.kW + 1
			input := randFloat32Slice(testCase.batch*testCase.inC*testCase.inH*testCase.inW, 0xC0)
			weight := randFloat32Slice(testCase.outC*testCase.inC*testCase.kH*testCase.kW, 0xC1)
			bias := randFloat32Slice(testCase.outC, 0xC2)
			got := make([]float32, testCase.batch*testCase.outC*outH*outW)
			want := make([]float32, len(got))

			convolution.Conv2DFloat32Native(
				config,
				convFloat32Pointer(input), convFloat32Pointer(weight),
				convFloat32Pointer(bias), convFloat32Pointer(got),
				testCase.batch, testCase.inC, testCase.inH, testCase.inW,
				testCase.outC, testCase.kH, testCase.kW, outH, outW,
			)
			convolution.Conv2DFloat32Scalar(
				config,
				convFloat32Pointer(input), convFloat32Pointer(weight),
				convFloat32Pointer(bias), convFloat32Pointer(want),
				testCase.batch, testCase.inC, testCase.inH, testCase.inW,
				testCase.outC, testCase.kH, testCase.kW, outH, outW,
			)

			assertFloat32SlicesNear(t, got, want, maxULPReduction)
		})
	}
}

func TestPool2DFloat32NEONParity(t *testing.T) {
	cases := []pool.PoolConfig{
		{KernelH: 3, KernelW: 3, StrideH: 1, StrideW: 1},
		pool.DefaultPoolConfig(),
	}

	for _, config := range cases {
		label := fmt.Sprintf("k=%dx%d_s=%dx%d", config.KernelH, config.KernelW, config.StrideH, config.StrideW)
		inH, inW := 16, 16
		outH := (inH-config.KernelH)/config.StrideH + 1
		outW := (inW-config.KernelW)/config.StrideW + 1
		input := randFloat32Slice(inH*inW, 0xA01)

		for _, useMax := range []bool{true, false} {
			poolLabel := label + "/max"

			if !useMax {
				poolLabel = label + "/avg"
			}

			t.Run(poolLabel, func(t *testing.T) {
				got := make([]float32, outH*outW)
				want := make([]float32, outH*outW)

				pool.Pool2DFloat32Native(
					config, convFloat32Pointer(input), convFloat32Pointer(got),
					1, 1, inH, inW, outH, outW,
					useMax,
				)
				pool.Pool2DFloat32Scalar(
					config, input, want,
					1, 1, inH, inW, outH, outW,
					useMax,
				)

				assertFloat32SlicesNear(t, got, want, maxULPElementary)
			})
		}
	}
}

func TestConv1DFloat32NEONParity(t *testing.T) {
	config := convolution.DefaultConv1DConfig()
	cases := []struct {
		batch, inC, inLen, outC, kLen int
	}{
		{1, 1, 8, 1, 3},
		{1, 3, 16, 2, 3},
		{2, 4, 32, 4, 5},
	}

	for _, testCase := range cases {
		label := fmt.Sprintf("b=%d_c=%d_len=%d", testCase.batch, testCase.inC, testCase.inLen)
		t.Run(label, func(t *testing.T) {
			outLen := testCase.inLen - testCase.kLen + 1
			input := randFloat32Slice(testCase.batch*testCase.inC*testCase.inLen, 0x1D0)
			weight := randFloat32Slice(testCase.outC*testCase.inC*testCase.kLen, 0x1D1)
			bias := randFloat32Slice(testCase.outC, 0x1D2)
			got := make([]float32, testCase.batch*testCase.outC*outLen)
			want := make([]float32, len(got))

			convolution.Conv1DFloat32Native(
				config,
				convFloat32Pointer(input), convFloat32Pointer(weight),
				convFloat32Pointer(bias), convFloat32Pointer(got),
				testCase.batch, testCase.inC, testCase.inLen,
				testCase.outC, testCase.kLen, outLen,
			)
			convolution.Conv1DFloat32Scalar(
				config,
				convFloat32Pointer(input), convFloat32Pointer(weight),
				convFloat32Pointer(bias), convFloat32Pointer(want),
				testCase.batch, testCase.inC, testCase.inLen,
				testCase.outC, testCase.kLen, outLen,
			)

			assertFloat32SlicesNear(t, got, want, maxULPReduction)
		})
	}
}

func TestConv3DFloat32NEONParity(t *testing.T) {
	config := convolution.DefaultConv3DConfig()
	batch, inC, inD, inH, inW := 1, 2, 4, 8, 8
	outC, kD, kH, kW := 2, 3, 3, 3
	outD := inD - kD + 1
	outH := inH - kH + 1
	outW := inW - kW + 1
	input := randFloat32Slice(batch*inC*inD*inH*inW, 0x3D0)
	weight := randFloat32Slice(outC*inC*kD*kH*kW, 0x3D1)
	bias := randFloat32Slice(outC, 0x3D2)
	got := make([]float32, batch*outC*outD*outH*outW)
	want := make([]float32, len(got))

	convolution.Conv3DFloat32Native(
		config,
		convFloat32Pointer(input), convFloat32Pointer(weight),
		convFloat32Pointer(bias), convFloat32Pointer(got),
		batch, inC, inD, inH, inW,
		outC, kD, kH, kW, outD, outH, outW,
	)
	convolution.Conv3DFloat32Scalar(
		config,
		convFloat32Pointer(input), convFloat32Pointer(weight),
		convFloat32Pointer(bias), convFloat32Pointer(want),
		batch, inC, inD, inH, inW,
		outC, kD, kH, kW, outD, outH, outW,
	)

	assertFloat32SlicesNear(t, got, want, maxULPReduction)
}

func TestDropoutFloat32NativeParity(t *testing.T) {
	for _, elementCount := range []int{1, 7, 64, 1024, 8192} {
		t.Run(fmt.Sprintf("N=%d", elementCount), func(t *testing.T) {
			src := randFloat32Slice(elementCount, 0xD00)
			got := make([]float32, elementCount)
			want := make([]float32, elementCount)
			SeedNEON := dropout.DropoutSeedState(0xD00)
			seedScalar := SeedNEON
			keepProb := float32(0.75)

			dropout.DropoutFloat32Native(got, src, &SeedNEON, keepProb)
			dropout.DropoutFloat32Native(want, src, &seedScalar, keepProb)

			assertFloat32SlicesNear(t, got, want, 0)
		})
	}
}

func TestHawkesLogLikelihoodNEONParity(t *testing.T) {
	for _, eventCount := range []int{1, 7, 64, 128} {
		t.Run(fmt.Sprintf("events=%d", eventCount), func(t *testing.T) {
			eventTimes := make([]float32, eventCount)

			for index := range eventTimes {
				eventTimes[index] = float32(index)*0.3 + 0.1
			}

			got := make([]float32, 1)
			want := make([]float32, 1)

			hawkes.HawkesLogLikelihoodNative(eventTimes, 10.0, 0.2, 0.5, 1.0, got)
			hawkes.HawkesLogLikelihoodScalar(eventTimes, 10.0, 0.2, 0.5, 1.0, want)

			assertFloat32SlicesNear(t, got, want, maxULPReduction)
		})
	}
}

func TestMarkovMutualInformationNEONParity(t *testing.T) {
	cases := []struct {
		xCount, yCount int
	}{
		{1, 1}, {2, 3}, {4, 4}, {8, 8},
	}

	for _, testCase := range cases {
		label := fmt.Sprintf("x=%d_y=%d", testCase.xCount, testCase.yCount)
		t.Run(label, func(t *testing.T) {
			joint := randFloat32Slice(testCase.xCount*testCase.yCount, 0x510)
			total := float32(0)

			for index := range joint {
				joint[index] = float32(math.Abs(float64(joint[index])))
				total += joint[index]
			}

			for index := range joint {
				joint[index] /= total
			}

			got := make([]float32, 1)
			want := make([]float32, 1)

			hawkes.MarkovMutualInformationNative(joint, testCase.xCount, testCase.yCount, got)
			hawkes.MarkovMutualInformationScalar(joint, testCase.xCount, testCase.yCount, want)

			assertFloat32SlicesNear(t, got, want, maxULPReduction)
		})
	}
}

func TestSparseCSRMatMulRowNEONAsmDirect(t *testing.T) {
	out := make([]float32, 1)
	right := make([]float32, 16)
	right[0] = 3

	matmul.SparseCSRMatMulRowSingleNzNEONAsm(&out[0], 2, &right[0], 1)

	if parity.Float32ULPDistance(out[0], 6) > 1 {
		t.Fatalf("single nz got=%g want=6 ulp=%d", out[0], parity.Float32ULPDistance(out[0], 6))
	}

	out[0] = 0
	values := make([]float32, 3)
	colIdx := make([]int32, 3)
	right = make([]float32, 16)
	values[0], values[1], values[2] = 2, 3, 4
	colIdx[0], colIdx[1], colIdx[2] = 0, 1, 2
	for index := range right {
		right[index] = float32(index + 1)
	}

	for nzIndex := 0; nzIndex < 3; nzIndex++ {
		denseRow := right[colIdx[nzIndex]:]
		matmul.SparseCSRMatMulRowSingleNzNEONAsm(&out[0], values[nzIndex], &denseRow[0], 1)
	}

	want := float32(2*1 + 3*2 + 4*3)
	if parity.Float32ULPDistance(out[0], want) > 1 {
		t.Fatalf("got=%g want=%g ulp=%d", out[0], want, parity.Float32ULPDistance(out[0], want))
	}
}

func TestSparseCSRMatMulFloat32NEONParity(t *testing.T) {
	for _, n := range paritySizes {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			rows, cols, inner := 8, n, 16
			if n > 1024 {
				rows, cols, inner = 64, 64, 64
			}

			rowPtr := make([]int32, rows+1)
			values := randFloat32Slice(rows*3, 0x5A0)
			colIdx := make([]int32, rows*3)
			right := randFloat32Slice(inner*cols, 0x5A1)

			for rowIndex := range rows {
				rowPtr[rowIndex] = int32(rowIndex * 3)

				for nzIndex := 0; nzIndex < 3; nzIndex++ {
					colIdx[rowIndex*3+nzIndex] = int32((rowIndex*3 + nzIndex) % inner)
				}
			}

			rowPtr[rows] = int32(rows * 3)
			got := make([]float32, rows*cols)
			want := make([]float32, rows*cols)

			matmul.SparseCSRMatMulFloat32Native(got, values, right, rowPtr, colIdx, rows, cols)
			matmul.SparseCSRMatMulFloat32Scalar(want, values, right, rowPtr, colIdx, rows, cols)

			assertFloat32SlicesNear(t, got, want, maxULPReduction)
		})
	}
}

func TestHawkesExpSumNEONAsmDirect(t *testing.T) {
	exponents := []float32{-0.1, -0.15, -0.2, -0.25}
	got := hawkes.HawkesExpSumNEONAsm(&exponents[0], 4)
	want := float32(0)

	for _, value := range exponents {
		want += float32(math.Exp(float64(value)))
	}

	if parity.Float32ULPDistance(got, want) > 4 {
		t.Fatalf("got=%g want=%g ulp=%d", got, want, parity.Float32ULPDistance(got, want))
	}
}

func TestHawkesIntensityNEONParity(t *testing.T) {
	for _, eventCount := range []int{1, 7, 64, 128} {
		t.Run(fmt.Sprintf("events=%d", eventCount), func(t *testing.T) {
			eventTimes := make([]float32, eventCount)
			queryTimes := make([]float32, eventCount)

			for index := range eventTimes {
				eventTimes[index] = float32(index) * 0.25
				queryTimes[index] = float32(index)*0.25 + 0.1
			}

			got := make([]float32, eventCount)
			want := make([]float32, eventCount)

			hawkes.HawkesIntensityNative(eventTimes, queryTimes, got, 0.1, 0.5, 1.0)
			hawkes.HawkesIntensityScalar(eventTimes, queryTimes, want, 0.1, 0.5, 1.0)

			assertFloat32SlicesNear(t, got, want, maxULPReduction)
		})
	}
}

func TestFlashAttentionRowNEONParity(t *testing.T) {
	for _, depth := range []int{1, 7, 64} {
		t.Run(fmt.Sprintf("depth=%d", depth), func(t *testing.T) {
			seqQ, seqK, valueDim := 8, 16, depth
			query := randFloat32Slice(seqQ*depth, 0xFA)
			key := randFloat32Slice(seqK*depth, 0xFB)
			value := randFloat32Slice(seqK*valueDim, 0xFC)
			got := make([]float32, seqQ*valueDim)
			want := make([]float32, seqQ*valueDim)
			scale := float32(1.0 / math.Sqrt(float64(depth)))

			for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
				attention.RunFlashAttentionRowNative(query, key, value, got, rowIndex, seqK, depth, valueDim, scale, false)
				runFlashAttentionRowScalar(query, key, value, want, rowIndex, seqK, depth, valueDim, scale, false)
			}

			assertFloat32SlicesNear(t, got, want, maxULPReduction)
		})
	}
}

func runFlashAttentionRowScalar(
	queryView, keyView, valueView, outView []float32,
	rowIndex, seqK, depth, valueDim int,
	scale float32,
	causal bool,
) {
	maxScore := float32(math.Inf(-1))
	normalizer := float32(0)
	accumulator := make([]float32, valueDim)

	for keyIndex := 0; keyIndex < seqK; keyIndex++ {
		if causal && keyIndex > rowIndex {
			continue
		}

		queryRow := queryView[rowIndex*depth : (rowIndex+1)*depth]
		keyRow := keyView[keyIndex*depth : (keyIndex+1)*depth]
		score := dotFloat32Scalar(queryRow, keyRow) * scale
		oldMax := maxScore

		if score > maxScore {
			maxScore = score
		}

		alpha := float32(math.Exp(float64(oldMax - maxScore)))
		shifted := float32(math.Exp(float64(score - maxScore)))
		normalizer = normalizer*alpha + shifted

		for dimIndex := 0; dimIndex < valueDim; dimIndex++ {
			accumulator[dimIndex] = accumulator[dimIndex]*alpha +
				shifted*valueView[keyIndex*valueDim+dimIndex]
		}
	}

	if normalizer == 0 {
		return
	}

	for dimIndex := 0; dimIndex < valueDim; dimIndex++ {
		outView[rowIndex*valueDim+dimIndex] = accumulator[dimIndex] / normalizer
	}
}

func dotFloat32Scalar(left, right []float32) float32 {
	var sum float32

	for index := range left {
		sum += left[index] * right[index]
	}

	return sum
}

func TestPoolWindowMaxFloat32NativeParity(t *testing.T) {
	channel := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	got := pool.PoolWindowMaxFloat32Native(channel, 3, 0, 2, 0, 2)
	want := pool.PoolWindowMaxScalar(channel, 3, 0, 2, 0, 2)

	assertFloat32SlicesNear(t, []float32{got}, []float32{want}, 1)
}

func TestConvTranspose2dStride1RowNEONAsmDirect(t *testing.T) {
	t.Run("asm_params_direct", func(t *testing.T) {
		input := []float32{1, 2, 3, 4}
		got := []float32{0, 0, 0, 0}

		convolution.ConvTranspose2dTapNEONAsm(
			&got[0],
			2,
			&input[0],
			4,
		)

		assertFloat32SlicesNear(t, got, []float32{2, 4, 6, 8}, 1)
	})

	t.Run("single_tap", func(t *testing.T) {
		input := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		weight := []float32{2}
		got := []float32{0.25, 0.5, 0.75, 1.0}
		want := []float32{2.25, 4.5, 6.75, 9.0}

		convolution.ConvTranspose2dStride1RowNEON(
			got,
			input,
			weight,
			4,
			1, 1, 1, 4,
			0, 0,
		)

		assertFloat32SlicesNear(t, got, want, 1)
	})
}

func TestConvTranspose2DFloat32NEONParity(t *testing.T) {
	config := convolution.DefaultConv2DConfig()
	batch, inC, inH, inW := 1, 2, 8, 8
	outC, kH, kW := 2, 3, 3
	outH := (inH-1)*config.StrideH + kH
	outW := (inW-1)*config.StrideW + kW
	input := randFloat32Slice(batch*inC*inH*inW, 0x720)
	weight := randFloat32Slice(inC*outC*kH*kW, 0x721)
	bias := randFloat32Slice(outC, 0x722)
	got := make([]float32, batch*outC*outH*outW)
	want := make([]float32, len(got))

	convolution.ConvTranspose2DFloat32Native(
		config,
		convFloat32Pointer(input), convFloat32Pointer(weight),
		convFloat32Pointer(bias), convFloat32Pointer(got),
		batch, inC, inH, inW, outC, kH, kW, outH, outW,
	)
	convolution.ConvTranspose2DFloat32Scalar(
		config,
		convFloat32Pointer(input), convFloat32Pointer(weight),
		convFloat32Pointer(bias), convFloat32Pointer(want),
		batch, inC, inH, inW, outC, kH, kW, outH, outW,
	)

	assertFloat32SlicesNear(t, got, want, maxULPReduction)
}

func TestAdaptivePool2DFloat32NEONParity(t *testing.T) {
	batch, channels, inH, inW := 1, 2, 7, 9
	outH, outW := 3, 4
	input := randFloat32Slice(batch*channels*inH*inW, 0xAD0)
	got := make([]float32, batch*channels*outH*outW)
	want := make([]float32, len(got))

	for _, useMax := range []bool{true, false} {
		label := "max"

		if !useMax {
			label = "avg"
		}

		t.Run(label, func(t *testing.T) {
			pool.AdaptivePool2DFloat32Native(
				convFloat32Pointer(input), convFloat32Pointer(got),
				batch, channels, inH, inW, outH, outW,
				useMax,
			)
			pool.AdaptivePool2DFloat32Scalar(
				input, want,
				batch, channels, inH, inW, outH, outW,
				useMax,
			)

			assertFloat32SlicesNear(t, got, want, maxULPReduction)
		})
	}
}

func TestConv2DGeneralFloat32NEONParity(t *testing.T) {
	config := convolution.Conv2DConfig{
		StrideH: 2, StrideW: 2,
		PaddingH: 1, PaddingW: 1,
		DilationH: 1, DilationW: 1,
	}
	batch, inC, inH, inW := 1, 2, 8, 8
	outC, kH, kW := 2, 3, 3
	outH := (inH+2*config.PaddingH-kH)/config.StrideH + 1
	outW := (inW+2*config.PaddingW-kW)/config.StrideW + 1
	input := randFloat32Slice(batch*inC*inH*inW, 0x2D0)
	weight := randFloat32Slice(outC*inC*kH*kW, 0x2D1)
	bias := randFloat32Slice(outC, 0x2D2)
	got := make([]float32, batch*outC*outH*outW)
	want := make([]float32, len(got))

	convolution.Conv2DFloat32Native(
		config,
		convFloat32Pointer(input), convFloat32Pointer(weight),
		convFloat32Pointer(bias), convFloat32Pointer(got),
		batch, inC, inH, inW, outC, kH, kW, outH, outW,
	)
	convolution.Conv2DFloat32Scalar(
		config,
		convFloat32Pointer(input), convFloat32Pointer(weight),
		convFloat32Pointer(bias), convFloat32Pointer(want),
		batch, inC, inH, inW, outC, kH, kW, outH, outW,
	)

	assertFloat32SlicesNear(t, got, want, maxULPReduction)
}

func TestPool2DGeneralFloat32NEONParity(t *testing.T) {
	config := pool.PoolConfig{
		KernelH: 3, KernelW: 3,
		StrideH: 2, StrideW: 2,
		PaddingH: 1, PaddingW: 1,
	}
	inH, inW := 8, 8
	outH := (inH+2*config.PaddingH-config.KernelH)/config.StrideH + 1
	outW := (inW+2*config.PaddingW-config.KernelW)/config.StrideW + 1
	input := randFloat32Slice(inH*inW, 0x520)

	for _, useMax := range []bool{true, false} {
		label := "max"

		if !useMax {
			label = "avg"
		}

		t.Run(label, func(t *testing.T) {
			got := make([]float32, outH*outW)
			want := make([]float32, outH*outW)

			pool.Pool2DFloat32Native(
				config, convFloat32Pointer(input), convFloat32Pointer(got),
				1, 1, inH, inW, outH, outW,
				useMax,
			)
			pool.Pool2DFloat32Scalar(
				config, input, want,
				1, 1, inH, inW, outH, outW,
				useMax,
			)

			assertFloat32SlicesNear(t, got, want, maxULPReduction)
		})
	}
}

func BenchmarkConv2DFloat32Native(b *testing.B) {
	config := convolution.DefaultConv2DConfig()
	input := randFloat32Slice(1*16*32*32, 0xBC0)
	weight := randFloat32Slice(8*16*3*3, 0xBC1)
	bias := randFloat32Slice(8, 0xBC2)
	output := make([]float32, 1*8*30*30)

	b.ResetTimer()

	for b.Loop() {
		convolution.Conv2DFloat32Native(
			config,
			convFloat32Pointer(input), convFloat32Pointer(weight),
			convFloat32Pointer(bias), convFloat32Pointer(output),
			1, 16, 32, 32,
			8, 3, 3, 30, 30,
		)
	}
}

func BenchmarkSparseCSRMatMulFloat32Native(b *testing.B) {
	rows, cols, inner := 64, 64, 64
	rowPtr := make([]int32, rows+1)
	values := randFloat32Slice(rows*4, 0x5A0)
	colIdx := make([]int32, rows*4)
	right := randFloat32Slice(inner*cols, 0x5A1)
	output := make([]float32, rows*cols)

	for rowIndex := range rows {
		rowPtr[rowIndex] = int32(rowIndex * 4)

		for nzIndex := 0; nzIndex < 4; nzIndex++ {
			colIdx[rowIndex*4+nzIndex] = int32((rowIndex*4 + nzIndex) % inner)
		}
	}

	rowPtr[rows] = int32(rows * 4)

	b.ResetTimer()

	for b.Loop() {
		matmul.SparseCSRMatMulFloat32Native(output, values, right, rowPtr, colIdx, rows, cols)
	}
}

func BenchmarkConvTranspose2DFloat32Native(b *testing.B) {
	config := convolution.DefaultConv2DConfig()
	batch, inC, inH, inW := 1, 8, 64, 64
	outC, kH, kW := 8, 3, 3
	outH := (inH-1)*config.StrideH + kH
	outW := (inW-1)*config.StrideW + kW
	input := randFloat32Slice(batch*inC*inH*inW, 0xB720)
	weight := randFloat32Slice(inC*outC*kH*kW, 0xB721)
	bias := randFloat32Slice(outC, 0xB722)
	output := make([]float32, batch*outC*outH*outW)

	b.ResetTimer()

	for b.Loop() {
		convolution.ConvTranspose2DFloat32Native(
			config,
			convFloat32Pointer(input), convFloat32Pointer(weight),
			convFloat32Pointer(bias), convFloat32Pointer(output),
			batch, inC, inH, inW, outC, kH, kW, outH, outW,
		)
	}
}

func BenchmarkAdaptivePool2DFloat32Native(b *testing.B) {
	batch, channels, inH, inW := 1, 16, 64, 64
	outH, outW := 8, 8
	input := randFloat32Slice(batch*channels*inH*inW, 0xBAD0)
	output := make([]float32, batch*channels*outH*outW)

	b.ResetTimer()

	for b.Loop() {
		pool.AdaptivePool2DFloat32Native(
			convFloat32Pointer(input), convFloat32Pointer(output),
			batch, channels, inH, inW, outH, outW,
			true,
		)
	}
}

func BenchmarkPool2DGeneralFloat32Native(b *testing.B) {
	config := pool.PoolConfig{KernelH: 3, KernelW: 3, StrideH: 2, StrideW: 2, PaddingH: 1, PaddingW: 1}
	batch, channels, inH, inW := 1, 16, 64, 64
	outH := (inH+2*config.PaddingH-config.KernelH)/config.StrideH + 1
	outW := (inW+2*config.PaddingW-config.KernelW)/config.StrideW + 1
	input := randFloat32Slice(batch*channels*inH*inW, 0xB0D0)
	output := make([]float32, batch*channels*outH*outW)

	b.ResetTimer()

	for b.Loop() {
		pool.Pool2DFloat32Native(
			config, convFloat32Pointer(input), convFloat32Pointer(output),
			batch, channels, inH, inW, outH, outW,
			true,
		)
	}
}

func BenchmarkFlashAttentionRowNative(b *testing.B) {
	seqQ, seqK, depth, valueDim := 64, 128, 64, 64
	query := randFloat32Slice(seqQ*depth, 0xFA0)
	key := randFloat32Slice(seqK*depth, 0xFB0)
	value := randFloat32Slice(seqK*valueDim, 0xFC0)
	output := make([]float32, seqQ*valueDim)
	scale := float32(1.0 / math.Sqrt(float64(depth)))

	b.ResetTimer()

	for b.Loop() {
		for rowIndex := range seqQ {
			attention.RunFlashAttentionRowNative(
				query, key, value, output,
				rowIndex, seqK, depth, valueDim, scale, false,
			)
		}
	}
}

func BenchmarkPoolWindowMaxFloat32Native(b *testing.B) {
	channel := randFloat32Slice(64*64, 0xB000)

	b.ResetTimer()

	for b.Loop() {
		_ = pool.PoolWindowMaxFloat32Native(channel, 64, 4, 12, 8, 20)
	}
}

func BenchmarkConvTranspose2dTapNEONAsm(b *testing.B) {
	input := randFloat32Slice(64*64, 0xC070)
	output := make([]float32, 64)
	weightVal := float32(0.5)

	b.ResetTimer()

	for b.Loop() {
		convolution.ConvTranspose2dTapNEONAsm(&output[0], weightVal, &input[32], 64)
	}
}
