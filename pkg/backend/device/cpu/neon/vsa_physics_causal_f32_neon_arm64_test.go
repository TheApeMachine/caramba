//go:build arm64

package neon

import (
	"fmt"
	"math"
	"testing"
)

var vsaPhysicsCausalSizes = []int{1, 7, 64, 1024, 8192}

func TestVSABindFloat32NEONParity(t *testing.T) {
	for _, size := range vsaPhysicsCausalSizes {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			left := randFloat32Slice(size, 0xB01)
			right := randFloat32Slice(size, 0xB02)
			got := make([]float32, size)
			want := make([]float32, size)

			VsaBindFloat32Native(got, left, right)
			for index := range want {
				want[index] = left[index] * right[index]
			}

			assertFloat32SlicesNear(t, got, want, 0)
		})
	}
}

func TestVSABundleFloat32NEONParity(t *testing.T) {
	for _, size := range vsaPhysicsCausalSizes {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			left := randFloat32Slice(size, 0xB03)
			right := randFloat32Slice(size, 0xB04)
			got := make([]float32, size)
			want := make([]float32, size)

			VsaBundleFloat32Native(got, left, right)
			for index := range want {
				want[index] = left[index] + right[index]
			}

			assertFloat32SlicesNear(t, got, want, 0)
		})
	}
}

func TestVSAPermuteFloat32NEONParity(t *testing.T) {
	for _, size := range vsaPhysicsCausalSizes {
		for _, shift := range []int{0, 1, 3, -2} {
			label := fmt.Sprintf("n=%d_shift=%d", size, shift)
			t.Run(label, func(t *testing.T) {
				src := randFloat32Slice(size, 0xB05)
				got := make([]float32, size)
				want := make([]float32, size)

				VsaPermuteFloat32Native(got, src, shift)
				scalarVSAPermute(want, src, shift)

				assertFloat32SlicesNear(t, got, want, 0)
			})
		}
	}
}

func TestVSASimilarityFloat32NEONParity(t *testing.T) {
	for _, size := range vsaPhysicsCausalSizes {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			left := randFloat32Slice(size, 0xB06)
			right := randFloat32Slice(size, 0xB07)

			got := VsaSimilarityFloat32Native(left, right)
			want := vsaDotFloat32Scalar(left, right)

			assertFloat32Near(t, got, want, 1e-5)
		})
	}
}

func TestLaplacian1DFloat32NEONParity(t *testing.T) {
	invH2 := float32(0.25)

	for _, size := range vsaPhysicsCausalSizes {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			input := randFloat32Slice(size, 0xC01)
			got := make([]float32, size)
			want := make([]float32, size)

			Laplacian1DFloat32Native(input, got, invH2)
			laplacian1D(input, want, size, 1.0/invH2)

			assertFloat32SlicesNear(t, got, want, 1e-5)
		})
	}
}

func TestGrad1DFloat32NEONParity(t *testing.T) {
	invTwoDx := float32(2.0)

	for _, size := range vsaPhysicsCausalSizes {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			input := randFloat32Slice(size, 0xC02)
			got := make([]float32, size)
			want := make([]float32, size)

			Grad1DFloat32Native(input, got, invTwoDx)
			for index := 0; index < size; index++ {
				left := input[(index-1+size)%size]
				right := input[(index+1)%size]
				want[index] = (right - left) * invTwoDx
			}

			assertFloat32SlicesNear(t, got, want, 1e-5)
		})
	}
}

func TestLaplacian4Float32NEONParity(t *testing.T) {
	invDen := float32(1.0 / 12.0)

	for _, size := range vsaPhysicsCausalSizes {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			input := randFloat32Slice(size, 0xC03)
			got := make([]float32, size)
			want := make([]float32, size)

			Laplacian4Float32Native(input, got, invDen)

			for index := 0; index < size; index++ {
				um2 := input[(index-2+size)%size]
				um1 := input[(index-1+size)%size]
				u0 := input[index]
				up1 := input[(index+1)%size]
				up2 := input[(index+2)%size]
				want[index] = (-um2 + 16*um1 - 30*u0 + 16*up1 - up2) * invDen
			}

			assertFloat32SlicesNear(t, got, want, 1e-4)
		})
	}
}

func TestCATEFloat32NEONParity(t *testing.T) {
	for _, size := range vsaPhysicsCausalSizes {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			treated := randFloat32Slice(size, 0xD01)
			control := randFloat32Slice(size, 0xD02)
			got := make([]float32, size)
			want := make([]float32, size)

			CateFloat32Native(treated, control, got)
			for index := range want {
				want[index] = treated[index] - control[index]
			}

			assertFloat32SlicesNear(t, got, want, 0)
		})
	}
}

func TestCounterfactualFloat32NEONParity(t *testing.T) {
	const slope = float32(1.75)

	for _, size := range vsaPhysicsCausalSizes {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			observedY := randFloat32Slice(size, 0xD03)
			observedX := randFloat32Slice(size, 0xD04)
			counterfactualX := randFloat32Slice(size, 0xD05)
			got := make([]float32, size)
			want := make([]float32, size)

			CounterfactualFloat32Native(got, observedY, observedX, counterfactualX, slope)
			for index := range want {
				want[index] = observedY[index] + slope*(counterfactualX[index]-observedX[index])
			}

			assertFloat32SlicesNear(t, got, want, 1e-6)
		})
	}
}

func TestBackdoorAdjustmentFloat32NEONParity(t *testing.T) {
	cases := []struct {
		xCount, zCount, yCount int
	}{
		{2, 3, 1},
		{3, 4, 2},
		{4, 7, 3},
	}

	for _, testCase := range cases {
		label := fmt.Sprintf("x=%d_z=%d_y=%d", testCase.xCount, testCase.zCount, testCase.yCount)
		t.Run(label, func(t *testing.T) {
			conditional := randFloat32Slice(testCase.xCount*testCase.zCount*testCase.yCount, 0xD06)
			marginalZ := randFloat32Slice(testCase.zCount, 0xD07)
			got := make([]float32, testCase.xCount*testCase.yCount)
			want := make([]float32, len(got))

			BackdoorAdjustmentFloat32Native(
				conditional, marginalZ, got,
				testCase.xCount, testCase.zCount, testCase.yCount,
			)
			scalarBackdoorAdjustment(
				conditional, marginalZ, want,
				testCase.xCount, testCase.zCount, testCase.yCount,
			)

			assertFloat32SlicesNear(t, got, want, 1e-6)
		})
	}
}

func TestIVEstimateFloat32NEONParity(t *testing.T) {
	for _, size := range []int{7, 64, 1024, 8192} {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			instrument := randFloat32Slice(size, 0xD08)
			treatment := randFloat32Slice(size, 0xD09)
			outcome := randFloat32Slice(size, 0xD0A)

			got := IvEstimateFloat32Native(instrument, treatment, outcome)
			want := ivEstimateFloat32Scalar(instrument, treatment, outcome)

			assertFloat32Near(t, got, want, 1e-4)
		})
	}
}

func TestFrontdoorAdjustmentFloat32NEONParity(t *testing.T) {
	cases := []struct {
		xCount, mCount, yCount int
	}{
		{2, 3, 1},
		{3, 4, 2},
		{4, 5, 3},
	}

	for _, testCase := range cases {
		label := fmt.Sprintf("x=%d_m=%d_y=%d", testCase.xCount, testCase.mCount, testCase.yCount)
		t.Run(label, func(t *testing.T) {
			mediatorGivenX := randFloat32Slice(testCase.xCount*testCase.mCount, 0xD0B)
			outcomeGivenXM := randFloat32Slice(testCase.xCount*testCase.mCount*testCase.yCount, 0xD0C)
			marginalX := randFloat32Slice(testCase.xCount, 0xD0D)
			got := make([]float32, testCase.xCount*testCase.yCount)
			want := make([]float32, len(got))

			FrontdoorAdjustmentFloat32Native(
				mediatorGivenX, outcomeGivenXM, marginalX, got,
				testCase.xCount, testCase.mCount, testCase.yCount,
			)
			scalarFrontdoorAdjustment(
				mediatorGivenX, outcomeGivenXM, marginalX, want,
				testCase.xCount, testCase.mCount, testCase.yCount,
			)

			assertFloat32SlicesNear(t, got, want, 1e-6)
		})
	}
}

func TestQuantumPotentialFloat32NEONParity(t *testing.T) {
	invH2 := float32(4.0)
	scale := float32(-0.5)

	for _, size := range vsaPhysicsCausalSizes {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			density := randFloat32Slice(size, 0xE10)
			got := make([]float32, size)
			want := make([]float32, size)

			QuantumPotentialFloat32Native(density, got, invH2, scale)
			scalarQuantumPotential(density, want, invH2, scale)

			assertFloat32SlicesNear(t, got, want, 1e-5)
		})
	}
}

func TestMadelungContinuityFloat32NEONParity(t *testing.T) {
	invTwoDx := float32(0.5)

	for _, size := range vsaPhysicsCausalSizes {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			density := randFloat32Slice(size, 0xE11)
			velocity := randFloat32Slice(size, 0xE12)
			got := make([]float32, size)
			want := make([]float32, size)

			MadelungContinuityFloat32Native(density, velocity, got, invTwoDx)
			scalarMadelungContinuity(density, velocity, want, invTwoDx)

			assertFloat32SlicesNear(t, got, want, 1e-5)
		})
	}
}

func TestLaplacian2DFloat32NEONParity(t *testing.T) {
	cases := []struct {
		rows, cols int
	}{
		{4, 4},
		{7, 8},
		{16, 16},
	}

	invH2 := float32(0.25)
	dxSquared := float32(1.0 / invH2)

	for _, testCase := range cases {
		label := fmt.Sprintf("rows=%d_cols=%d", testCase.rows, testCase.cols)
		t.Run(label, func(t *testing.T) {
			elementCount := testCase.rows * testCase.cols
			input := randFloat32Slice(elementCount, 0xE13)
			got := make([]float32, elementCount)
			want := make([]float32, elementCount)
			scratch := make([]float32, testCase.rows*2)

			Laplacian2DFloat32Native(input, got, scratch, testCase.rows, testCase.cols, invH2)
			laplacian2D(input, want, testCase.rows, testCase.cols, dxSquared)

			assertFloat32SlicesNear(t, got, want, 1e-4)
		})
	}
}

func scalarVSAPermute(dst, src []float32, shift int) {
	elementCount := len(src)

	if elementCount == 0 {
		return
	}

	rotation := ((shift % elementCount) + elementCount) % elementCount

	for index, value := range src {
		target := (index + rotation) % elementCount
		dst[target] = value
	}
}

func scalarBackdoorAdjustment(
	conditional, marginalZ, out []float32,
	xCount, zCount, yCount int,
) {
	for index := range out {
		out[index] = 0
	}

	for xIndex := 0; xIndex < xCount; xIndex++ {
		for yIndex := 0; yIndex < yCount; yIndex++ {
			var sum float32

			for zIndex := 0; zIndex < zCount; zIndex++ {
				condIndex := (xIndex*zCount+zIndex)*yCount + yIndex
				sum += conditional[condIndex] * marginalZ[zIndex]
			}

			out[xIndex*yCount+yIndex] = sum
		}
	}
}

func scalarFrontdoorAdjustment(
	mediatorGivenX, outcomeGivenXM, marginalX, out []float32,
	xCount, mCount, yCount int,
) {
	for index := range out {
		out[index] = 0
	}

	for xIndex := 0; xIndex < xCount; xIndex++ {
		for yIndex := 0; yIndex < yCount; yIndex++ {
			var total float32

			for mIndex := 0; mIndex < mCount; mIndex++ {
				pmx := mediatorGivenX[xIndex*mCount+mIndex]

				var innerSum float32

				for xPrimeIndex := 0; xPrimeIndex < xCount; xPrimeIndex++ {
					outcomeIndex := (xPrimeIndex*mCount+mIndex)*yCount + yIndex
					innerSum += outcomeGivenXM[outcomeIndex] * marginalX[xPrimeIndex]
				}

				total += pmx * innerSum
			}

			out[xIndex*yCount+yIndex] = total
		}
	}
}

func scalarQuantumPotential(density, out []float32, invH2, scale float32) {
	elementCount := len(density)

	if elementCount == 0 {
		return
	}

	out[0] = 0
	out[elementCount-1] = 0

	if elementCount <= 2 {
		return
	}

	const eps = float32(1e-12)

	for index := 1; index < elementCount-1; index++ {
		rho := density[index]

		if rho <= eps {
			out[index] = 0
			continue
		}

		sqrtRho := float32(math.Sqrt(float64(rho)))
		sqrtLeft := float32(math.Sqrt(float64(math.Max(float64(eps), float64(density[index-1])))))
		sqrtRight := float32(math.Sqrt(float64(math.Max(float64(eps), float64(density[index+1])))))
		laplacian := (sqrtRight - 2*sqrtRho + sqrtLeft) * invH2
		out[index] = scale * laplacian / sqrtRho
	}
}

func scalarMadelungContinuity(density, velocity, out []float32, invTwoDx float32) {
	elementCount := len(density)

	if elementCount == 0 {
		return
	}

	out[0] = 0
	out[elementCount-1] = 0

	if elementCount <= 2 {
		return
	}

	for index := 1; index < elementCount-1; index++ {
		fluxRight := density[index+1] * velocity[index+1]
		fluxLeft := density[index-1] * velocity[index-1]
		out[index] = (fluxRight - fluxLeft) * invTwoDx
	}
}

func vsaDotFloat32Scalar(left, right []float32) float32 {
	var sum float32

	for index := range left {
		sum += left[index] * right[index]
	}

	return sum
}

func ivEstimateFloat32Scalar(instrument, treatment, outcome []float32) float32 {
	elementCount := len(instrument)
	var meanZ, meanX, meanY float64

	for index := 0; index < elementCount; index++ {
		meanZ += float64(instrument[index])
		meanX += float64(treatment[index])
		meanY += float64(outcome[index])
	}

	meanZ /= float64(elementCount)
	meanX /= float64(elementCount)
	meanY /= float64(elementCount)

	var covZY, covZX float64

	for index := 0; index < elementCount; index++ {
		deltaZ := float64(instrument[index]) - meanZ
		deltaY := float64(outcome[index]) - meanY
		deltaX := float64(treatment[index]) - meanX
		covZY += deltaZ * deltaY
		covZX += deltaZ * deltaX
	}

	if math.Abs(covZX) < 1e-12 {
		return 0
	}

	return float32(covZY / covZX)
}

func assertFloat32Near(t *testing.T, got, want float32, tolerance float64) {
	t.Helper()

	if math.Abs(float64(got-want)) > tolerance {
		t.Fatalf("got %g want %g tolerance %g", got, want, tolerance)
	}
}

func BenchmarkVSABindFloat32Native(b *testing.B) {
	size := 8192
	left := randFloat32Slice(size, 0xE01)
	right := randFloat32Slice(size, 0xE02)
	dst := make([]float32, size)

	for b.Loop() {
		VsaBindFloat32Native(dst, left, right)
	}
}

func BenchmarkLaplacian1DFloat32Native(b *testing.B) {
	size := 8192
	input := randFloat32Slice(size, 0xE03)
	out := make([]float32, size)
	invH2 := float32(0.25)

	for b.Loop() {
		Laplacian1DFloat32Native(input, out, invH2)
	}
}

func BenchmarkCounterfactualFloat32Native(b *testing.B) {
	size := 8192
	observedY := randFloat32Slice(size, 0xE04)
	observedX := randFloat32Slice(size, 0xE05)
	counterfactualX := randFloat32Slice(size, 0xE06)
	out := make([]float32, size)

	for b.Loop() {
		CounterfactualFloat32Native(out, observedY, observedX, counterfactualX, 1.5)
	}
}

func BenchmarkQuantumPotentialFloat32Native(b *testing.B) {
	size := 8192
	density := randFloat32Slice(size, 0xE20)
	out := make([]float32, size)
	invH2 := float32(4.0)
	scale := float32(-0.5)

	for b.Loop() {
		QuantumPotentialFloat32Native(density, out, invH2, scale)
	}
}

func BenchmarkMadelungContinuityFloat32Native(b *testing.B) {
	size := 8192
	density := randFloat32Slice(size, 0xE21)
	velocity := randFloat32Slice(size, 0xE22)
	out := make([]float32, size)
	invTwoDx := float32(0.5)

	for b.Loop() {
		MadelungContinuityFloat32Native(density, velocity, out, invTwoDx)
	}
}

func BenchmarkFrontdoorAdjustmentFloat32Native(b *testing.B) {
	mediatorGivenX := randFloat32Slice(4*5, 0xE23)
	outcomeGivenXM := randFloat32Slice(4*5*3, 0xE24)
	marginalX := randFloat32Slice(4, 0xE25)
	out := make([]float32, 4*3)

	for b.Loop() {
		FrontdoorAdjustmentFloat32Native(
			mediatorGivenX, outcomeGivenXM, marginalX, out,
			4, 5, 3,
		)
	}
}
