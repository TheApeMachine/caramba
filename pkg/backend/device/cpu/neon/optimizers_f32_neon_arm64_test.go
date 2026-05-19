//go:build arm64

package neon

import (
	"fmt"
	"math"
	"testing"
)

/*
Optimizer NEON parity against the scalar reference at N ∈ {1, 7, 64,
1024, 8192} per AGENTS.md §2 and §6.
*/

func TestAdamStepSlicesNEONParity(t *testing.T) {
	for _, n := range paritySizes {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			config := DefaultAdamConfig()
			params := randFloat32Slice(n, 0xAD)
			grad := randFloat32Slice(n, 0xBE)
			FirstNEON := randFloat32Slice(n, 0xC0)
			SecondNEON := randFloat32Slice(n, 0xC1)
			firstScalar := append([]float32(nil), FirstNEON...)
			secondScalar := append([]float32(nil), SecondNEON...)
			OutNEON := make([]float32, n)
			outScalar := make([]float32, n)

			adamStepSlices(config, params, grad, FirstNEON, SecondNEON, OutNEON)
			adamStepSlicesScalar(config, params, grad, firstScalar, secondScalar, outScalar)

			assertFloat32SlicesEqual(t, OutNEON, outScalar)
			assertFloat32SlicesEqual(t, FirstNEON, firstScalar)
			assertFloat32SlicesEqual(t, SecondNEON, secondScalar)
		})
	}
}

func TestAdamWStepSlicesNEONParity(t *testing.T) {
	runTwoStateOptimizerParity(t, DefaultAdamWConfig(),
		adamWStepSlices, adamWStepSlicesScalar,
	)
}

func TestSGDStepSlicesNEONParity(t *testing.T) {
	runOneStateOptimizerParity(t, DefaultSGDConfig(),
		sgdStepSlices, sgdStepSlicesScalar,
	)
}

func TestAdamaxStepSlicesNEONParity(t *testing.T) {
	runTwoStateOptimizerParity(t, DefaultAdamaxConfig(),
		adamaxStepSlices, adamaxStepSlicesScalar,
	)
}

func TestAdagradStepSlicesNEONParity(t *testing.T) {
	runOneStateOptimizerParity(t, DefaultAdagradConfig(),
		adagradStepSlices, adagradStepSlicesScalar,
	)
}

func TestRMSpropStepSlicesNEONParity(t *testing.T) {
	runOneStateOptimizerParity(t, DefaultRMSpropConfig(),
		rmspropStepSlices, rmspropStepSlicesScalar,
	)
}

func TestLionStepSlicesNEONParity(t *testing.T) {
	runOneStateOptimizerParity(t, DefaultLionConfig(),
		lionStepSlices, lionStepSlicesScalar,
	)
}

func TestLARSStepSlicesNEONParity(t *testing.T) {
	runOneStateOptimizerParity(t, DefaultLARSConfig(),
		larsStepSlices, larsStepSlicesScalar,
	)
}

func TestLBFGSStepSlicesNEONParity(t *testing.T) {
	for _, n := range paritySizes {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			config := DefaultLBFGSConfig()
			params := randFloat32Slice(n, 0x1B)
			grad := randFloat32Slice(n, 0xF6)
			OutNEON := make([]float32, n)
			outScalar := make([]float32, n)

			lbfgsStepSlices(config, params, grad, OutNEON)
			lbfgsStepSlicesScalar(config, params, grad, outScalar)

			assertFloat32SlicesEqual(t, OutNEON, outScalar)
		})
	}
}

func TestHebbianStepSlicesNEONParity(t *testing.T) {
	cases := []struct {
		postDim int
		preDim  int
	}{
		{1, 1}, {1, 7}, {7, 64}, {64, 64}, {8, 1024},
	}

	for _, testCase := range cases {
		label := fmt.Sprintf("post=%d_pre=%d", testCase.postDim, testCase.preDim)

		t.Run(label, func(t *testing.T) {
			config := DefaultHebbianConfig()
			n := testCase.postDim * testCase.preDim
			weights := randFloat32Slice(n, 0xE00)
			post := randFloat32Slice(testCase.postDim, 0xBB0)
			pre := randFloat32Slice(testCase.preDim, 0xCC0)
			OutNEON := make([]float32, n)
			outScalar := make([]float32, n)

			hebbianStepSlices(config, weights, post, pre, OutNEON, testCase.preDim)
			hebbianStepSlicesScalar(config, weights, post, pre, outScalar, testCase.preDim)

			assertFloat32SlicesEqual(t, OutNEON, outScalar)
		})
	}
}

var paritySizes = []int{1, 7, 64, 1024, 8192}

func runOneStateOptimizerParity[C any](
	t *testing.T,
	config C,
	RunNEON func(C, []float32, []float32, []float32, []float32),
	runScalar func(C, []float32, []float32, []float32, []float32),
) {
	for _, n := range paritySizes {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			params := randFloat32Slice(n, 0xAD)
			grad := randFloat32Slice(n, 0xBE)
			StateNEON := randFloat32Slice(n, 0xC0)
			stateScalar := append([]float32(nil), StateNEON...)
			OutNEON := make([]float32, n)
			outScalar := make([]float32, n)

			RunNEON(config, params, grad, StateNEON, OutNEON)
			runScalar(config, params, grad, stateScalar, outScalar)

			assertFloat32SlicesEqual(t, OutNEON, outScalar)
			assertFloat32SlicesEqual(t, StateNEON, stateScalar)
		})
	}
}

func runTwoStateOptimizerParity[C any](
	t *testing.T,
	config C,
	RunNEON func(C, []float32, []float32, []float32, []float32, []float32),
	runScalar func(C, []float32, []float32, []float32, []float32, []float32),
) {
	for _, n := range paritySizes {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			params := randFloat32Slice(n, 0xAD)
			grad := randFloat32Slice(n, 0xBE)
			FirstNEON := randFloat32Slice(n, 0xC0)
			SecondNEON := randFloat32Slice(n, 0xC1)
			firstScalar := append([]float32(nil), FirstNEON...)
			secondScalar := append([]float32(nil), SecondNEON...)
			OutNEON := make([]float32, n)
			outScalar := make([]float32, n)

			RunNEON(config, params, grad, FirstNEON, SecondNEON, OutNEON)
			runScalar(config, params, grad, firstScalar, secondScalar, outScalar)

			assertFloat32SlicesEqual(t, OutNEON, outScalar)
			assertFloat32SlicesEqual(t, FirstNEON, firstScalar)
			assertFloat32SlicesEqual(t, SecondNEON, secondScalar)
		})
	}
}

func assertFloat32SlicesEqual(t *testing.T, got, want []float32) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("length mismatch got=%d want=%d", len(got), len(want))
	}

	for index := range got {
		if got[index] == want[index] {
			continue
		}

		if math.IsNaN(float64(got[index])) && math.IsNaN(float64(want[index])) {
			continue
		}

		if float32ULPDistance(got[index], want[index]) <= 1 {
			continue
		}

		t.Fatalf("lane %d got=%g want=%g ulp=%d",
			index, got[index], want[index],
			float32ULPDistance(got[index], want[index]),
		)
	}
}

func float32ULPDistance(left, right float32) int {
	leftBits := math.Float32bits(left)
	rightBits := math.Float32bits(right)

	if leftBits > rightBits {
		leftBits, rightBits = rightBits, leftBits
	}

	return int(rightBits - leftBits)
}
