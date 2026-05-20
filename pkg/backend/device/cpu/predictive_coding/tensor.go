package predictive_coding

import "github.com/theapemachine/manifesto/tensor"

func RunPCPrediction(args ...tensor.Tensor) error {
	return runPCPrediction(args...)
}

func RunPCPredictionError(args ...tensor.Tensor) error {
	return runPCPredictionError(args...)
}

func RunPCUpdateRepresentationDefault(args ...tensor.Tensor) error {
	return runPCUpdateRepresentationDefault(args...)
}

func RunPCUpdateWeightsDefault(args ...tensor.Tensor) error {
	return runPCUpdateWeightsDefault(args...)
}
