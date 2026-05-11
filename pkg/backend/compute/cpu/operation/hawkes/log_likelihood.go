package hawkes

import (
	"fmt"
	stdmath "math"
)

/*
LogLikelihood computes the Hawkes process log-likelihood:

  L = Σ_i log λ(t_i) - ∫_0^{T_end} λ(t) dt

For exponential kernel with base rate mu:
  ∫ λ(t) dt = mu * T_end + (alpha/beta) * Σ_i (1 - exp(-beta*(T_end - t_i)))

shape = [T, T_end_idx]  (T_end_idx is shape[1], the index sentinel)
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

	ll := applyLogLikelihood(times, intensities, baselineIntegral[0], T)

	return []float64{ll}
}

func applyLogLikelihood(times, intensities []float64, integral float64, T int) float64 {
	sumLog := applyLogLikelihoodSumLog(intensities, T)
	return sumLog - integral
}

func applyLogLikelihoodScalar(intensities []float64, T int) float64 {
	sum := 0.0

	for idx := range T {
		lam := intensities[idx]

		if lam > 0 {
			sum += stdmath.Log(lam)
		}
	}

	return sum
}
