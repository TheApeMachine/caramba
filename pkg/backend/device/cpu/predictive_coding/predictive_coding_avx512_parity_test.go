//go:build amd64

package predictive_coding

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512PredictiveCodingAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestPCPredictionF32AVX512Parity(t *testing.T) {
	if !avx512PredictiveCodingAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given PCPredictionF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PredictionFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xB311+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xB312+int64(length))
				got := make([]float32, outDim)
				want := make([]float32, outDim)

				PCPredictionF32AVX512(
					&weights[0], &representation[0], &got[0], outDim, inDim,
				)
				PredictionFloat32Scalar(weights, representation, want, outDim, inDim)

				assertPredictionParity(t, got, want, outDim)
			})
		}
	})
}

func TestPCPredictionErrorF32AVX512Parity(t *testing.T) {
	if !avx512PredictiveCodingAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given PCPredictionErrorF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PredictionErrorFloat32Scalar for N=%d", length), func() {
				observed, predicted, _ := randomPredictiveCodingVectors(length, 0xB313+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				PCPredictionErrorF32AVX512(
					&observed[0], &predicted[0], &got[0], length,
				)
				PredictionErrorFloat32Scalar(observed, predicted, want)

				assertSliceParity(t, got, want)
			})
		}
	})
}

func TestPCUpdateRepresentationF32AVX512Parity(t *testing.T) {
	if !avx512PredictiveCodingAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given PCUpdateRepresentationF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match UpdateRepresentationFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xB314+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xB315+int64(length))
				_, predictionError, _ := randomPredictiveCodingVectors(outDim, 0xB315+int64(length)+1)
				got := make([]float32, inDim)
				want := make([]float32, inDim)

				PCUpdateRepresentationF32AVX512(
					&weights[0], &representation[0], &predictionError[0], &got[0],
					predictiveCodingTestLR, outDim, inDim,
				)
				UpdateRepresentationFloat32Scalar(
					weights, representation, predictionError, want,
					predictiveCodingTestLR, outDim, inDim,
				)

				assertSliceParity(t, got, want)
			})
		}
	})
}

func TestPCUpdateWeightsF32AVX512Parity(t *testing.T) {
	if !avx512PredictiveCodingAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given PCUpdateWeightsF32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match UpdateWeightsFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xB316+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xB317+int64(length))
				_, predictionError, _ := randomPredictiveCodingVectors(outDim, 0xB317+int64(length)+1)
				got := make([]float32, outDim*inDim)
				want := make([]float32, outDim*inDim)

				PCUpdateWeightsF32AVX512(
					&weights[0], &representation[0], &predictionError[0], &got[0],
					predictiveCodingTestLR, outDim, inDim,
				)
				UpdateWeightsFloat32Scalar(
					weights, representation, predictionError, want,
					predictiveCodingTestLR, outDim, inDim,
				)

				assertSliceParity(t, got, want)
			})
		}
	})
}
