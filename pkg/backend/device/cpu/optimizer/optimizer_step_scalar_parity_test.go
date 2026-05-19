package optimizer

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

/*
OptimizerStepScalarParity exercises scalar reference paths at parity.Lengths on the host.
*/

func TestAdamStepSlicesScalarParityLengths(t *testing.T) {
	runTwoStateOptimizerScalarParity(t, DefaultAdamConfig(), adamStepSlicesScalar)
}

func TestAdamWStepSlicesScalarParityLengths(t *testing.T) {
	runTwoStateOptimizerScalarParity(t, DefaultAdamWConfig(), adamWStepSlicesScalar)
}

func TestSGDStepSlicesScalarParityLengths(t *testing.T) {
	runOneStateOptimizerScalarParity(t, DefaultSGDConfig(), sgdStepSlicesScalar)
}

func TestAdamaxStepSlicesScalarParityLengths(t *testing.T) {
	runTwoStateOptimizerScalarParity(t, DefaultAdamaxConfig(), adamaxStepSlicesScalar)
}

func TestAdagradStepSlicesScalarParityLengths(t *testing.T) {
	runOneStateOptimizerScalarParity(t, DefaultAdagradConfig(), adagradStepSlicesScalar)
}

func TestRMSpropStepSlicesScalarParityLengths(t *testing.T) {
	runOneStateOptimizerScalarParity(t, DefaultRMSpropConfig(), rmspropStepSlicesScalar)
}

func TestLionStepSlicesScalarParityLengths(t *testing.T) {
	runOneStateOptimizerScalarParity(t, DefaultLionConfig(), lionStepSlicesScalar)
}

func TestLARSStepSlicesScalarParityLengths(t *testing.T) {
	runOneStateOptimizerScalarParity(t, DefaultLARSConfig(), larsStepSlicesScalar)
}

func TestLBFGSStepSlicesScalarParityLengths(t *testing.T) {
	convey.Convey("Given lbfgsStepSlicesScalar", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should be deterministic for N=%d", length), func() {
				config := DefaultLBFGSConfig()
				params := randFloat32Slice(length, 0x25B0+int64(length))
				grad := randFloat32Slice(length, 0x25B1+int64(length))
				outFirst := make([]float32, length)
				outSecond := make([]float32, length)

				lbfgsStepSlicesScalar(config, params, grad, outFirst)
				lbfgsStepSlicesScalar(config, params, grad, outSecond)

				parity.AssertFloat32SlicesWithinULP(t, outFirst, outSecond, 0)
			})
		}
	})
}

func TestHebbianStepSlicesScalarParityLengths(t *testing.T) {
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
			weights := randFloat32Slice(elementCount, 0x25E0)
			post := randFloat32Slice(testCase.postDim, 0x25E1)
			pre := randFloat32Slice(testCase.preDim, 0x25E2)
			outFirst := make([]float32, elementCount)
			outSecond := make([]float32, elementCount)

			hebbianStepSlicesScalar(config, weights, post, pre, outFirst, testCase.preDim)
			hebbianStepSlicesScalar(config, weights, post, pre, outSecond, testCase.preDim)

			parity.AssertFloat32SlicesWithinULP(t, outFirst, outSecond, 0)
		})
	}
}

func runOneStateOptimizerScalarParity[C any](
	t *testing.T,
	config C,
	runScalar func(C, []float32, []float32, []float32, []float32),
) {
	convey.Convey("Given optimizer scalar step slices", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should be deterministic for N=%d", length), func() {
				params := randFloat32Slice(length, 0x2600+int64(length))
				grad := randFloat32Slice(length, 0x2601+int64(length))
				stateFirst := randFloat32Slice(length, 0x2602+int64(length))
				stateSecond := append([]float32(nil), stateFirst...)
				outFirst := make([]float32, length)
				outSecond := make([]float32, length)

				runScalar(config, params, grad, stateFirst, outFirst)
				runScalar(config, params, grad, stateSecond, outSecond)

				parity.AssertFloat32SlicesWithinULP(t, outFirst, outSecond, 0)
				parity.AssertFloat32SlicesWithinULP(t, stateFirst, stateSecond, 0)
			})
		}
	})
}

func runTwoStateOptimizerScalarParity[C any](
	t *testing.T,
	config C,
	runScalar func(C, []float32, []float32, []float32, []float32, []float32),
) {
	convey.Convey("Given optimizer scalar two-state step slices", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should be deterministic for N=%d", length), func() {
				params := randFloat32Slice(length, 0x2700+int64(length))
				grad := randFloat32Slice(length, 0x2701+int64(length))
				firstFirst := randFloat32Slice(length, 0x2702+int64(length))
				secondFirst := randFloat32Slice(length, 0x2703+int64(length))
				firstSecond := append([]float32(nil), firstFirst...)
				secondSecond := append([]float32(nil), secondFirst...)
				outFirst := make([]float32, length)
				outSecond := make([]float32, length)

				runScalar(config, params, grad, firstFirst, secondFirst, outFirst)
				runScalar(config, params, grad, firstSecond, secondSecond, outSecond)

				parity.AssertFloat32SlicesWithinULP(t, outFirst, outSecond, 0)
				parity.AssertFloat32SlicesWithinULP(t, firstFirst, firstSecond, 0)
				parity.AssertFloat32SlicesWithinULP(t, secondFirst, secondSecond, 0)
			})
		}
	})
}
