package predictive_coding

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
Predictive-coding primitives — the four-kernel loop of the canonical
Friston-style hierarchical predictive coding model:

  - prediction:           top-down prediction p̂ = W × s
  - prediction_error:     e = observed - p̂
  - update_representation: s ← s + lr × W^T × e
  - update_weights:       W ← W + lr × outer(e, s)

Host tensor paths route through Float32Native dispatchers (AVX-512 on amd64 when available).
*/

type PredictiveCodingConfig struct {
	LearningRate float32
}

func DefaultPredictiveCodingConfig() PredictiveCodingConfig {
	return PredictiveCodingConfig{LearningRate: 1e-2}
}

/*
runPCPrediction computes p̂ = W × s. Args: (weights [out, in],
representation [in], output [out]).
*/
func runPCPrediction(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	wView, _ := args[0].Float32Native()
	sView, _ := args[1].Float32Native()
	outView, _ := args[2].Float32Native()

	wDims := args[0].Shape().Dims()

	if len(wDims) != 2 || wDims[1] != len(sView) || len(outView) != wDims[0] {
		return tensor.ErrShapeMismatch
	}

	PredictionFloat32Native(wView, sView, outView, wDims[0], wDims[1])

	return nil
}

func runPCPredictionError(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	observed, _ := args[0].Float32Native()
	predicted, _ := args[1].Float32Native()
	out, _ := args[2].Float32Native()

	if len(observed) != len(predicted) || len(out) != len(observed) {
		return tensor.ErrShapeMismatch
	}

	PredictionErrorFloat32Native(observed, predicted, out)

	return nil
}

func runPCUpdateRepresentationDefault(args ...tensor.Tensor) error {
	return PCUpdateRepresentation(
		DefaultPredictiveCodingConfig(),
		args[0], args[1], args[2], args[3],
	)
}

/*
PCUpdateRepresentation: s_new = s + lr × W^T × e. Args:
(weights [out, in], representation [in], error [out], output [in]).
*/
func PCUpdateRepresentation(
	config PredictiveCodingConfig,
	weights, representation, predictionError, output tensor.Tensor,
) error {
	wView, _ := weights.Float32Native()
	sView, _ := representation.Float32Native()
	eView, _ := predictionError.Float32Native()
	outView, _ := output.Float32Native()

	wDims := weights.Shape().Dims()

	if len(wDims) != 2 ||
		wDims[1] != len(sView) ||
		wDims[0] != len(eView) ||
		len(outView) != len(sView) {
		return tensor.ErrShapeMismatch
	}

	UpdateRepresentationFloat32Native(
		wView, sView, eView, outView,
		config.LearningRate, wDims[0], wDims[1],
	)

	return nil
}

func runPCUpdateWeightsDefault(args ...tensor.Tensor) error {
	return PCUpdateWeights(
		DefaultPredictiveCodingConfig(),
		args[0], args[1], args[2], args[3],
	)
}

/*
PCUpdateWeights: W_new = W + lr × outer(e, s).
*/
func PCUpdateWeights(
	config PredictiveCodingConfig,
	weights, representation, predictionError, output tensor.Tensor,
) error {
	wView, _ := weights.Float32Native()
	sView, _ := representation.Float32Native()
	eView, _ := predictionError.Float32Native()
	outView, _ := output.Float32Native()

	wDims := weights.Shape().Dims()

	if len(wDims) != 2 ||
		wDims[1] != len(sView) ||
		wDims[0] != len(eView) ||
		len(outView) != len(wView) {
		return tensor.ErrShapeMismatch
	}

	UpdateWeightsFloat32Native(
		wView, sView, eView, outView,
		config.LearningRate, wDims[0], wDims[1],
	)

	return nil
}
