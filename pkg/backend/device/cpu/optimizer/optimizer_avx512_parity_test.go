//go:build amd64

package optimizer

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const optimizerAVX512MaxULP = 2

func avx512OptimizerAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestAdamStepSlicesAVX512Parity(t *testing.T) {
	if !avx512OptimizerAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given AdamStepSlicesAVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match adamStepSlicesScalar for N=%d", length), func() {
				config := DefaultAdamConfig()
				params := randFloat32Slice(length, 0x2100+int64(length))
				grad := randFloat32Slice(length, 0x2101+int64(length))
				firstAVX := randFloat32Slice(length, 0x2102+int64(length))
				secondAVX := randFloat32Slice(length, 0x2103+int64(length))
				firstScalar := append([]float32(nil), firstAVX...)
				secondScalar := append([]float32(nil), secondAVX...)
				outAVX := make([]float32, length)
				outScalar := make([]float32, length)

				adamStepSlicesAVX512(config, params, grad, firstAVX, secondAVX, outAVX)
				adamStepSlicesScalar(config, params, grad, firstScalar, secondScalar, outScalar)

				parity.AssertFloat32SlicesWithinULP(t, outAVX, outScalar, optimizerAVX512MaxULP)
				parity.AssertFloat32SlicesWithinULP(t, firstAVX, firstScalar, optimizerAVX512MaxULP)
				parity.AssertFloat32SlicesWithinULP(t, secondAVX, secondScalar, optimizerAVX512MaxULP)
			})
		}

		convey.Convey("It should match adamStepSlicesScalar via direct asm at parity.Lengths", func() {
			config := DefaultAdamConfig()
			beta1Correction := 1 - float32(math.Pow(float64(config.Beta1), float64(config.Step)))
			beta2Correction := 1 - float32(math.Pow(float64(config.Beta2), float64(config.Step)))

			for _, length := range parity.Lengths {
				params := randFloat32Slice(length, 0x2110+int64(length))
				grad := randFloat32Slice(length, 0x2111+int64(length))
				firstAVX := randFloat32Slice(length, 0x2112+int64(length))
				secondAVX := randFloat32Slice(length, 0x2113+int64(length))
				firstScalar := append([]float32(nil), firstAVX...)
				secondScalar := append([]float32(nil), secondAVX...)
				outAVX := make([]float32, length)
				outScalar := make([]float32, length)

				AdamStepFloat32AVX512Asm(
					&params[0], &grad[0], &firstAVX[0], &secondAVX[0], &outAVX[0],
					length,
					config.LearningRate, config.Beta1, config.Beta2, config.Epsilon,
					beta1Correction, beta2Correction,
				)
				adamStepSlicesScalar(config, params, grad, firstScalar, secondScalar, outScalar)

				parity.AssertFloat32SlicesWithinULP(t, outAVX, outScalar, optimizerAVX512MaxULP)
				parity.AssertFloat32SlicesWithinULP(t, firstAVX, firstScalar, optimizerAVX512MaxULP)
				parity.AssertFloat32SlicesWithinULP(t, secondAVX, secondScalar, optimizerAVX512MaxULP)
			}
		})
	})
}

func TestAdamWStepSlicesAVX512Parity(t *testing.T) {
	if !avx512OptimizerAvailable() {
		t.Skip("AVX-512F required")
	}

	runTwoStateOptimizerAVX512Parity(t, DefaultAdamWConfig(), adamWStepSlicesAVX512, adamWStepSlicesScalar)
}

func TestSGDStepSlicesAVX512Parity(t *testing.T) {
	if !avx512OptimizerAvailable() {
		t.Skip("AVX-512F required")
	}

	runOneStateOptimizerAVX512Parity(t, DefaultSGDConfig(), sgdStepSlicesAVX512, sgdStepSlicesScalar)
}

func TestAdamaxStepSlicesAVX512Parity(t *testing.T) {
	if !avx512OptimizerAvailable() {
		t.Skip("AVX-512F required")
	}

	runTwoStateOptimizerAVX512Parity(t, DefaultAdamaxConfig(), adamaxStepSlicesAVX512, adamaxStepSlicesScalar)
}

func TestAdagradStepSlicesAVX512Parity(t *testing.T) {
	if !avx512OptimizerAvailable() {
		t.Skip("AVX-512F required")
	}

	runOneStateOptimizerAVX512Parity(t, DefaultAdagradConfig(), adagradStepSlicesAVX512, adagradStepSlicesScalar)
}

func TestRMSpropStepSlicesAVX512Parity(t *testing.T) {
	if !avx512OptimizerAvailable() {
		t.Skip("AVX-512F required")
	}

	runOneStateOptimizerAVX512Parity(t, DefaultRMSpropConfig(), rmspropStepSlicesAVX512, rmspropStepSlicesScalar)
}

func TestLionStepSlicesAVX512Parity(t *testing.T) {
	if !avx512OptimizerAvailable() {
		t.Skip("AVX-512F required")
	}

	runOneStateOptimizerAVX512Parity(t, DefaultLionConfig(), lionStepSlicesAVX512, lionStepSlicesScalar)
}

func TestLARSStepSlicesAVX512Parity(t *testing.T) {
	if !avx512OptimizerAvailable() {
		t.Skip("AVX-512F required")
	}

	runOneStateOptimizerAVX512Parity(t, DefaultLARSConfig(), larsStepSlicesAVX512, larsStepSlicesScalar)
}

func TestLBFGSStepSlicesAVX512Parity(t *testing.T) {
	if !avx512OptimizerAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given LBFGSStepSlicesAVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match lbfgsStepSlicesScalar for N=%d", length), func() {
				config := DefaultLBFGSConfig()
				params := randFloat32Slice(length, 0x21B0+int64(length))
				grad := randFloat32Slice(length, 0x21B1+int64(length))
				outAVX := make([]float32, length)
				outScalar := make([]float32, length)

				lbfgsStepSlicesAVX512(config, params, grad, outAVX)
				lbfgsStepSlicesScalar(config, params, grad, outScalar)

				parity.AssertFloat32SlicesWithinULP(t, outAVX, outScalar, optimizerAVX512MaxULP)
			})
		}
	})
}

func TestHebbianStepSlicesAVX512Parity(t *testing.T) {
	if !avx512OptimizerAvailable() {
		t.Skip("AVX-512F required")
	}

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
			elementCount := testCase.postDim * testCase.preDim
			weights := randFloat32Slice(elementCount, 0x21E0)
			post := randFloat32Slice(testCase.postDim, 0x21E1)
			pre := randFloat32Slice(testCase.preDim, 0x21E2)
			outAVX := make([]float32, elementCount)
			outScalar := make([]float32, elementCount)

			hebbianStepSlicesAVX512(config, weights, post, pre, outAVX, testCase.preDim)
			hebbianStepSlicesScalar(config, weights, post, pre, outScalar, testCase.preDim)

			parity.AssertFloat32SlicesWithinULP(t, outAVX, outScalar, optimizerAVX512MaxULP)
		})
	}
}

func runOneStateOptimizerAVX512Parity[C any](
	t *testing.T,
	config C,
	runAVX512 func(C, []float32, []float32, []float32, []float32),
	runScalar func(C, []float32, []float32, []float32, []float32),
) {
	convey.Convey("Given optimizer AVX-512 step slices", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar for N=%d", length), func() {
				params := randFloat32Slice(length, 0x2200+int64(length))
				grad := randFloat32Slice(length, 0x2201+int64(length))
				stateAVX := randFloat32Slice(length, 0x2202+int64(length))
				stateScalar := append([]float32(nil), stateAVX...)
				outAVX := make([]float32, length)
				outScalar := make([]float32, length)

				runAVX512(config, params, grad, stateAVX, outAVX)
				runScalar(config, params, grad, stateScalar, outScalar)

				parity.AssertFloat32SlicesWithinULP(t, outAVX, outScalar, optimizerAVX512MaxULP)
				parity.AssertFloat32SlicesWithinULP(t, stateAVX, stateScalar, optimizerAVX512MaxULP)
			})
		}
	})
}

func runTwoStateOptimizerAVX512Parity[C any](
	t *testing.T,
	config C,
	runAVX512 func(C, []float32, []float32, []float32, []float32, []float32),
	runScalar func(C, []float32, []float32, []float32, []float32, []float32),
) {
	convey.Convey("Given optimizer AVX-512 two-state step slices", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match scalar for N=%d", length), func() {
				params := randFloat32Slice(length, 0x2300+int64(length))
				grad := randFloat32Slice(length, 0x2301+int64(length))
				firstAVX := randFloat32Slice(length, 0x2302+int64(length))
				secondAVX := randFloat32Slice(length, 0x2303+int64(length))
				firstScalar := append([]float32(nil), firstAVX...)
				secondScalar := append([]float32(nil), secondAVX...)
				outAVX := make([]float32, length)
				outScalar := make([]float32, length)

				runAVX512(config, params, grad, firstAVX, secondAVX, outAVX)
				runScalar(config, params, grad, firstScalar, secondScalar, outScalar)

				parity.AssertFloat32SlicesWithinULP(t, outAVX, outScalar, optimizerAVX512MaxULP)
				parity.AssertFloat32SlicesWithinULP(t, firstAVX, firstScalar, optimizerAVX512MaxULP)
				parity.AssertFloat32SlicesWithinULP(t, secondAVX, secondScalar, optimizerAVX512MaxULP)
			})
		}
	})
}
