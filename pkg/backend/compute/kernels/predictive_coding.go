package kernels

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Predictive-coding primitives — the four-kernel loop of the canonical
Friston-style hierarchical predictive coding model:

  - prediction:           top-down prediction p̂ = W × s
  - prediction_error:     e = observed - p̂
  - update_representation: s ← s + lr × W^T × e
  - update_weights:       W ← W + lr × outer(e, s)
*/

type PredictiveCodingConfig struct {
	LearningRate float32
}

func DefaultPredictiveCodingConfig() PredictiveCodingConfig {
	return PredictiveCodingConfig{LearningRate: 1e-2}
}

func init() {
	Default.Register(Kernel{
		Name: "pc_prediction",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runPCPrediction,
	})

	Default.Register(Kernel{
		Name: "pc_prediction_error",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runPCPredictionError,
	})

	Default.Register(Kernel{
		Name: "pc_update_representation",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runPCUpdateRepresentationDefault,
	})

	Default.Register(Kernel{
		Name: "pc_update_weights",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runPCUpdateWeightsDefault,
	})
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

	for outIndex := 0; outIndex < wDims[0]; outIndex++ {
		var sum float32

		for inIndex := 0; inIndex < wDims[1]; inIndex++ {
			sum += wView[outIndex*wDims[1]+inIndex] * sView[inIndex]
		}

		outView[outIndex] = sum
	}

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

	for index, value := range observed {
		out[index] = value - predicted[index]
	}

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

	copy(outView, sView)

	for outIndex := 0; outIndex < wDims[0]; outIndex++ {
		for inIndex := 0; inIndex < wDims[1]; inIndex++ {
			outView[inIndex] += config.LearningRate *
				wView[outIndex*wDims[1]+inIndex] * eView[outIndex]
		}
	}

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

	copy(outView, wView)

	for outIndex := 0; outIndex < wDims[0]; outIndex++ {
		for inIndex := 0; inIndex < wDims[1]; inIndex++ {
			outView[outIndex*wDims[1]+inIndex] +=
				config.LearningRate * eView[outIndex] * sView[inIndex]
		}
	}

	return nil
}
