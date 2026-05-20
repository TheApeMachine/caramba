package hawkes

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func HawkesIntensityScalar(
	eventTimes, queryTimes, out []float32,
	mu, alpha, beta float32,
) {
	for queryIndex, queryTime := range queryTimes {
		intensity := mu

		for _, eventTime := range eventTimes {
			if eventTime > queryTime {
				continue
			}

			intensity += alpha * hawkesExpScalar(-beta*(queryTime-eventTime))
		}

		out[queryIndex] = intensity
	}
}

func HawkesLogLikelihoodScalar(
	eventTimes []float32,
	totalT, mu, alpha, beta float32,
	out []float32,
) {
	var sumLog float64

	for eventIndex, eventTime := range eventTimes {
		intensity := mu

		for previousIndex := 0; previousIndex < eventIndex; previousIndex++ {
			delta := eventTime - eventTimes[previousIndex]
			intensity += alpha * hawkesExpScalar(-beta*delta)
		}

		sumLog += math.Log(math.Max(1e-12, float64(intensity)))
	}

	compensator := float64(mu * totalT)

	for _, eventTime := range eventTimes {
		compensator += float64(alpha/beta) * (1 - math.Exp(float64(-beta*(totalT-eventTime))))
	}

	out[0] = float32(sumLog - compensator)
}

func MarkovMutualInformationScalar(joint []float32, xCount, yCount int, out []float32) {
	marginalX := make([]float64, xCount)
	marginalY := make([]float64, yCount)

	for xIndex := 0; xIndex < xCount; xIndex++ {
		for yIndex := 0; yIndex < yCount; yIndex++ {
			value := float64(joint[xIndex*yCount+yIndex])
			marginalX[xIndex] += value
			marginalY[yIndex] += value
		}
	}

	const eps = 1e-12
	var mutualInformation float64

	for xIndex := 0; xIndex < xCount; xIndex++ {
		for yIndex := 0; yIndex < yCount; yIndex++ {
			jointValue := float64(joint[xIndex*yCount+yIndex])

			if jointValue <= eps {
				continue
			}

			mutualInformation += jointValue * math.Log(jointValue/(marginalX[xIndex]*marginalY[yIndex]+eps))
		}
	}

	out[0] = float32(mutualInformation)
}

func markovBlanketPartition(
	adjacency []float32,
	internal []int32,
	out []int32,
	nodeCount int,
) {
	isInternal := make([]bool, nodeCount)

	for _, nodeID := range internal {
		if int(nodeID) >= 0 && int(nodeID) < nodeCount {
			isInternal[int(nodeID)] = true
		}
	}

	for nodeID := 0; nodeID < nodeCount; nodeID++ {
		if isInternal[nodeID] {
			out[nodeID] = 0
			continue
		}

		hasIncomingFromInternal := false
		hasOutgoingToInternal := false

		for otherID := 0; otherID < nodeCount; otherID++ {
			if !isInternal[otherID] {
				continue
			}

			if adjacency[otherID*nodeCount+nodeID] != 0 {
				hasIncomingFromInternal = true
			}

			if adjacency[nodeID*nodeCount+otherID] != 0 {
				hasOutgoingToInternal = true
			}
		}

		switch {
		case hasIncomingFromInternal && hasOutgoingToInternal:
			out[nodeID] = 2
		case hasOutgoingToInternal:
			out[nodeID] = 1
		default:
			out[nodeID] = 3
		}
	}
}

func RunHawkesIntensity(args ...tensor.Tensor) error {
	if len(args) != 6 {
		return tensor.ErrShapeMismatch
	}

	eventTimes, _ := args[0].Float32Native()
	queryTimes, _ := args[1].Float32Native()
	baseline, _ := args[2].Float32Native()
	alpha, _ := args[3].Float32Native()
	beta, _ := args[4].Float32Native()
	out, _ := args[5].Float32Native()

	if len(baseline) < 1 || len(alpha) < 1 || len(beta) < 1 ||
		len(out) != len(queryTimes) {
		return tensor.ErrShapeMismatch
	}

	HawkesIntensityNative(eventTimes, queryTimes, out, baseline[0], alpha[0], beta[0])

	return nil
}

func RunHawkesKernelMatrix(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	eventTimes, _ := args[0].Float32Native()
	alpha, _ := args[1].Float32Native()
	beta, _ := args[2].Float32Native()
	out, _ := args[3].Float32Native()

	eventCount := len(eventTimes)

	if len(alpha) < 1 || len(beta) < 1 || len(out) != eventCount*eventCount {
		return tensor.ErrShapeMismatch
	}

	HawkesKernelMatrixNative(eventTimes, out, alpha[0], beta[0])

	return nil
}

func RunHawkesLogLikelihood(args ...tensor.Tensor) error {
	if len(args) != 6 {
		return tensor.ErrShapeMismatch
	}

	eventTimes, _ := args[0].Float32Native()
	totalT, _ := args[1].Float32Native()
	baseline, _ := args[2].Float32Native()
	alpha, _ := args[3].Float32Native()
	beta, _ := args[4].Float32Native()
	out, _ := args[5].Float32Native()

	if len(totalT) < 1 || len(baseline) < 1 ||
		len(alpha) < 1 || len(beta) < 1 || len(out) < 1 {
		return tensor.ErrShapeMismatch
	}

	HawkesLogLikelihoodNative(eventTimes, totalT[0], baseline[0], alpha[0], beta[0], out)

	return nil
}

func RunMarkovMutualInformation(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	joint, _ := args[0].Float32Native()
	out, _ := args[1].Float32Native()

	dims := args[0].Shape().Dims()

	if len(dims) != 2 || len(out) < 1 {
		return tensor.ErrShapeMismatch
	}

	MarkovMutualInformationNative(joint, dims[0], dims[1], out)

	return nil
}

func RunMarkovBlanketPartition(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	adjacency, _ := args[0].Float32Native()
	internal, _ := args[1].Int32Native()
	out, _ := args[2].Int32Native()

	dims := args[0].Shape().Dims()

	if len(dims) != 2 || dims[0] != dims[1] || len(out) != dims[0] {
		return tensor.ErrShapeMismatch
	}

	markovBlanketPartition(adjacency, internal, out, dims[0])

	return nil
}
