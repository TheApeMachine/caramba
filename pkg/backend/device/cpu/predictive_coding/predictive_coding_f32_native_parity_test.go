package predictive_coding

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
)

func TestPredictionFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given PredictionFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PredictionFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xB301+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xB302+int64(length))
				got := make([]float32, outDim)
				want := make([]float32, outDim)

				PredictionFloat32Native(weights, representation, got, outDim, inDim)
				PredictionFloat32Scalar(weights, representation, want, outDim, inDim)

				assertPredictionParity(t, got, want, outDim)
			})
		}
	})
}

func TestPredictionErrorFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given PredictionErrorFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PredictionErrorFloat32Scalar for N=%d", length), func() {
				observed, predicted, _ := randomPredictiveCodingVectors(length, 0xB303+int64(length))
				got := make([]float32, length)
				want := make([]float32, length)

				PredictionErrorFloat32Native(observed, predicted, got)
				PredictionErrorFloat32Scalar(observed, predicted, want)

				assertSliceParity(t, got, want)
			})
		}
	})
}

func TestUpdateRepresentationFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given UpdateRepresentationFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match UpdateRepresentationFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xB304+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xB305+int64(length))
				_, predictionError, _ := randomPredictiveCodingVectors(outDim, 0xB305+int64(length)+1)
				got := make([]float32, inDim)
				want := make([]float32, inDim)

				UpdateRepresentationFloat32Native(
					weights, representation, predictionError, got,
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

func TestUpdateWeightsFloat32NativeParityLengths(t *testing.T) {
	convey.Convey("Given UpdateWeightsFloat32Native", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match UpdateWeightsFloat32Scalar for N=%d", length), func() {
				outDim, inDim := predictiveCodingParityDims(length)
				weights := randomPredictiveCodingWeights(outDim, inDim, 0xB306+int64(length))
				_, representation, _ := randomPredictiveCodingVectors(inDim, 0xB307+int64(length))
				_, predictionError, _ := randomPredictiveCodingVectors(outDim, 0xB307+int64(length)+1)
				got := make([]float32, outDim*inDim)
				want := make([]float32, outDim*inDim)

				UpdateWeightsFloat32Native(
					weights, representation, predictionError, got,
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
