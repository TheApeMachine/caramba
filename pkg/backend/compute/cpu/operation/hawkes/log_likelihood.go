package hawkes

import (
	"fmt"
	"math"

	mathops "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
LogLikelihood computes the Hawkes process log-likelihood:

	L = Σ_i log λ(t_i) - ∫_0^{T_end} λ(t) dt

For exponential kernel with base rate mu:

	∫ λ(t) dt = mu * T_end + (alpha/beta) * Σ_i (1 - exp(-beta*(T_end - t_i)))

shape = [T]   (event count; integral carries ∫λ dt separately in data[2])
data[0] = times [T]
data[1] = intensities_at_events [T]   — pre-computed λ(t_i) for each event
data[2] = baseline_integral [1]       — mu * T_end + (alpha/beta)*... pre-computed

Returns scalar log-likelihood as []float64{ll}.
*/
type LogLikelihood struct{}

func NewLogLikelihood() *LogLikelihood { return &LogLikelihood{} }

func (logLikelihood *LogLikelihood) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 1 {
		return nil, fmt.Errorf("hawkes.log_likelihood: len(shape)=%d, need >= 1", len(shape))
	}

	if err := stateDict.RequireOperationInputs("hawkes.log_likelihood", 3); err != nil {
		return nil, err
	}

	eventCount := shape[0]
	times := stateDict.Inputs[0]
	intensities := stateDict.Inputs[1]
	baselineIntegral := stateDict.Inputs[2]

	if eventCount < 0 {
		return nil, fmt.Errorf("hawkes.log_likelihood: T=%d, need T >= 0", eventCount)
	}

	if len(times) != eventCount {
		return nil, fmt.Errorf(
			"hawkes.log_likelihood: len(times)=%d, need T=%d",
			len(times), eventCount,
		)
	}

	if len(intensities) != eventCount {
		return nil, fmt.Errorf(
			"hawkes.log_likelihood: len(intensities)=%d, need T=%d",
			len(intensities), eventCount,
		)
	}

	if len(baselineIntegral) < 1 {
		return nil, fmt.Errorf("hawkes.log_likelihood: baseline integral must have length >= 1")
	}

	logLikelihoodValue := applyLogLikelihood(intensities, baselineIntegral[0], eventCount)
	stateDict.SetOperationOutput([]float64{logLikelihoodValue})

	return stateDict, nil
}

func applyLogLikelihood(intensities []float64, integral float64, T int) float64 {
	if T == 0 {
		return -integral
	}

	for idx := 0; idx < T; idx++ {
		if intensities[idx] <= 0 {
			return math.NaN()
		}
	}

	logs := make([]float64, T)
	mathops.LogVec(logs, intensities[:T])
	sumLog := mathops.ReduceSum(logs)

	return sumLog - integral
}

func applyLogLikelihoodScalar(intensities []float64, T int) float64 {
	sum := 0.0

	for idx := 0; idx < T; idx++ {
		lam := intensities[idx]

		if lam <= 0 {
			return math.NaN()
		}

		sum += math.Log(lam)
	}

	return sum
}
