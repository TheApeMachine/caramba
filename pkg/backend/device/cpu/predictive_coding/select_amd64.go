//go:build amd64

package predictive_coding

import "golang.org/x/sys/cpu"

func PredictionFloat32Native(
	weights []float32,
	representation, output []float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		PCPredictionF32AVX512(
			&weights[0], &representation[0], &output[0], outDim, inDim,
		)
		return
	}

	PredictionFloat32Scalar(weights, representation, output, outDim, inDim)
}

func PredictionErrorFloat32Native(observed, predicted, output []float32) {
	if len(observed) == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		PCPredictionErrorF32AVX512(
			&observed[0], &predicted[0], &output[0], len(observed),
		)
		return
	}

	PredictionErrorFloat32Scalar(observed, predicted, output)
}

func UpdateRepresentationFloat32Native(
	weights []float32,
	representation, predictionError, output []float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		PCUpdateRepresentationF32AVX512(
			&weights[0], &representation[0], &predictionError[0], &output[0],
			learningRate, outDim, inDim,
		)
		return
	}

	UpdateRepresentationFloat32Scalar(
		weights, representation, predictionError, output,
		learningRate, outDim, inDim,
	)
}

func UpdateWeightsFloat32Native(
	weights []float32,
	representation, predictionError, output []float32,
	learningRate float32,
	outDim, inDim int,
) {
	if outDim == 0 || inDim == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		PCUpdateWeightsF32AVX512(
			&weights[0], &representation[0], &predictionError[0], &output[0],
			learningRate, outDim, inDim,
		)
		return
	}

	UpdateWeightsFloat32Scalar(
		weights, representation, predictionError, output,
		learningRate, outDim, inDim,
	)
}
