package hawkes

import (
	"fmt"
	"math"

	mathops "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
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

func (op *LogLikelihood) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 1 {
		panic(fmt.Errorf("hawkes: LogLikelihood: len(shape)=%d, need >= 1", len(shape)).Error())
	}

	if len(data) < 3 {
		panic(fmt.Errorf("hawkes: LogLikelihood: len(data)=%d, need 3", len(data)).Error())
	}

	T := shape[0]
	times := data[0]
	intensities := data[1]
	baselineIntegral := data[2]

	if len(times) != T {
		panic(fmt.Errorf(
			"hawkes: LogLikelihood: len(times)=%d, need T=%d",
			len(times), T,
		).Error())
	}

	if len(intensities) != T {
		panic(fmt.Errorf(
			"hawkes: LogLikelihood: len(intensities)=%d, need T=%d",
			len(intensities), T,
		).Error())
	}

	if len(baselineIntegral) < 1 {
		panic(fmt.Errorf("hawkes: LogLikelihood: data[2] must have length >= 1").Error())
	}

	ll := applyLogLikelihood(intensities, baselineIntegral[0], T)

	return []float64{ll}
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
