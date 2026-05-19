package neon

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Hawkes process and Markov-blanket primitives.

Hawkes:
  - hawkes_intensity: λ(t) = μ + Σ α × exp(-β (t - t_i)).
  - hawkes_kernel_matrix: pairwise excitation between events.
  - hawkes_log_likelihood: log-likelihood of an event sequence.

Markov blanket:
  - markov_partition: split a [N, N] adjacency into internal /
    sensory / active / external Markov-blanket categories.
  - markov_mutual_information: I(X; Y) from a joint distribution.
*/

func init() {
	Default.Register(Kernel{
		Name: "hawkes_intensity",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runHawkesIntensity,
	})

	Default.Register(Kernel{
		Name: "hawkes_kernel_matrix",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runHawkesKernelMatrix,
	})

	Default.Register(Kernel{
		Name: "hawkes_log_likelihood",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runHawkesLogLikelihood,
	})

	Default.Register(Kernel{
		Name: "markov_mutual_information",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMarkovMutualInformation,
	})

	Default.Register(Kernel{
		Name: "markov_blanket_partition",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Int32},
			Outputs: []dtype.DType{dtype.Int32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMarkovBlanketPartition,
	})
}

/*
runHawkesIntensity computes λ(t) = baseline + Σ_i alpha × exp(-beta × (t - t_i))
for each output time. Args: (eventTimes, queryTimes, baseline_scalar,
alpha_scalar, beta_scalar, output).
*/
func runHawkesIntensity(args ...tensor.Tensor) error {
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

	mu := baseline[0]
	a := alpha[0]
	b := beta[0]

	HawkesIntensityNative(eventTimes, queryTimes, out, mu, a, b)

	return nil
}

func hawkesIntensityScalar(
	eventTimes, queryTimes, out []float32,
	mu, alpha, beta float32,
) {
	for queryIndex, queryTime := range queryTimes {
		intensity := mu

		for _, eventTime := range eventTimes {
			if eventTime > queryTime {
				continue
			}

			intensity += alpha * float32(math.Exp(float64(-beta*(queryTime-eventTime))))
		}

		out[queryIndex] = intensity
	}
}

/*
runHawkesKernelMatrix produces a [len(events), len(events)] matrix
of pairwise excitations K[i, j] = alpha × exp(-beta × (t_i - t_j))
for j < i, 0 otherwise.
*/
func runHawkesKernelMatrix(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	eventTimes, _ := args[0].Float32Native()
	alpha, _ := args[1].Float32Native()
	beta, _ := args[2].Float32Native()
	out, _ := args[3].Float32Native()

	n := len(eventTimes)

	if len(alpha) < 1 || len(beta) < 1 || len(out) != n*n {
		return tensor.ErrShapeMismatch
	}

	a := alpha[0]
	b := beta[0]

	HawkesKernelMatrixNative(eventTimes, out, a, b)

	return nil
}

func hawkesKernelMatrixScalar(
	eventTimes, out []float32,
	alpha, beta float32,
) {
	eventCount := len(eventTimes)

	for rowIndex := 0; rowIndex < eventCount; rowIndex++ {
		for colIndex := 0; colIndex < eventCount; colIndex++ {
			if colIndex >= rowIndex {
				out[rowIndex*eventCount+colIndex] = 0
				continue
			}

			delta := eventTimes[rowIndex] - eventTimes[colIndex]
			out[rowIndex*eventCount+colIndex] = alpha * float32(math.Exp(float64(-beta*delta)))
		}
	}
}

/*
runHawkesLogLikelihood computes log L = Σ ln λ(t_i) - ∫λ(t)dt over
[0, T]. Args: (eventTimes, totalT_scalar, baseline, alpha, beta,
output_scalar).
*/
func runHawkesLogLikelihood(args ...tensor.Tensor) error {
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

	mu := baseline[0]
	a := alpha[0]
	b := beta[0]
	t := totalT[0]

	HawkesLogLikelihoodNative(eventTimes, t, mu, a, b, out)

	return nil
}

func hawkesLogLikelihoodScalar(
	eventTimes []float32,
	totalT, mu, alpha, beta float32,
	out []float32,
) {
	var sumLog float64

	for eventIndex, eventTime := range eventTimes {
		intensity := mu

		for previousIndex := 0; previousIndex < eventIndex; previousIndex++ {
			delta := eventTime - eventTimes[previousIndex]
			intensity += alpha * float32(math.Exp(float64(-beta*delta)))
		}

		sumLog += math.Log(math.Max(1e-12, float64(intensity)))
	}

	compensator := float64(mu * totalT)

	for _, eventTime := range eventTimes {
		compensator += float64(alpha/beta) * (1 - math.Exp(float64(-beta*(totalT-eventTime))))
	}

	out[0] = float32(sumLog - compensator)
}

/*
runMarkovMutualInformation computes I(X; Y) from a joint distribution
table [|X|, |Y|]. Output is a scalar.
*/
func runMarkovMutualInformation(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	joint, _ := args[0].Float32Native()
	out, _ := args[1].Float32Native()

	dims := args[0].Shape().Dims()

	if len(dims) != 2 || len(out) < 1 {
		return tensor.ErrShapeMismatch
	}

	xCount := dims[0]
	yCount := dims[1]

	MarkovMutualInformationNative(joint, xCount, yCount, out)

	return nil
}

func markovMutualInformationScalar(joint []float32, xCount, yCount int, out []float32) {
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
	var mi float64

	for xIndex := 0; xIndex < xCount; xIndex++ {
		for yIndex := 0; yIndex < yCount; yIndex++ {
			pxy := float64(joint[xIndex*yCount+yIndex])

			if pxy <= eps {
				continue
			}

			mi += pxy * math.Log(pxy/(marginalX[xIndex]*marginalY[yIndex]+eps))
		}
	}

	out[0] = float32(mi)
}

/*
runMarkovBlanketPartition labels each node as 0=internal,
1=sensory, 2=active, 3=external based on adjacency and a list of
internal nodes. Args: (adjacency [N,N], internalNodes [k], output_labels [N]).
*/
func runMarkovBlanketPartition(args ...tensor.Tensor) error {
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

	n := dims[0]
	isInternal := make([]bool, n)

	for _, nodeID := range internal {
		if int(nodeID) >= 0 && int(nodeID) < n {
			isInternal[int(nodeID)] = true
		}
	}

	for nodeID := 0; nodeID < n; nodeID++ {
		if isInternal[nodeID] {
			out[nodeID] = 0
			continue
		}

		hasIncomingFromInternal := false
		hasOutgoingToInternal := false

		for otherID := 0; otherID < n; otherID++ {
			if !isInternal[otherID] {
				continue
			}

			if adjacency[otherID*n+nodeID] != 0 {
				hasIncomingFromInternal = true
			}

			if adjacency[nodeID*n+otherID] != 0 {
				hasOutgoingToInternal = true
			}
		}

		switch {
		case hasIncomingFromInternal && hasOutgoingToInternal:
			out[nodeID] = 2 // active
		case hasOutgoingToInternal:
			out[nodeID] = 1 // sensory
		default:
			out[nodeID] = 3 // external
		}
	}

	return nil
}
