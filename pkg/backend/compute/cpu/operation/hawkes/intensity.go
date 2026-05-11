package hawkes

import (
	"fmt"
	stdmath "math"
)

/*
Intensity computes the current Hawkes process intensity for K processes:

  λ_k(t) = mu_k + alpha_k * Σ_{t_i < t} exp(-beta_k * (t - t_i))

shape = [K, T]
data[0] = event_times [T]   (sorted ascending, all < current time t)
data[1] = alpha [K]
data[2] = beta  [K]
data[3] = mu    [K]
data[4] = t     [1]         (current time)

Returns λ [K].
*/
type Intensity struct{}

func NewIntensity() *Intensity { return &Intensity{} }

func (op *Intensity) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Errorf("hawkes: Intensity: len(shape)=%d, need >= 2", len(shape)).Error())
	}

	if len(data) < 5 {
		panic(fmt.Errorf("hawkes: Intensity: len(data)=%d, need 5", len(data)).Error())
	}

	K, T := shape[0], shape[1]
	times := data[0]
	alpha := data[1]
	beta := data[2]
	mu := data[3]
	tSlice := data[4]

	if len(times) != T {
		panic(fmt.Errorf("hawkes: Intensity: len(times)=%d, need T=%d", len(times), T).Error())
	}

	if len(alpha) != K || len(beta) != K || len(mu) != K {
		panic(fmt.Errorf(
			"hawkes: Intensity: alpha/beta/mu lengths must equal K=%d",
			K,
		).Error())
	}

	if len(tSlice) < 1 {
		panic(fmt.Errorf("hawkes: Intensity: data[4] (t) must have length >= 1").Error())
	}

	t := tSlice[0]
	out := make([]float64, K)
	applyIntensity(out, times, alpha, beta, mu, t, K, T)

	return out
}

func applyIntensityScalar(out, times, alpha, beta, mu []float64, t float64, K, T int) {
	for k := range K {
		sum := 0.0
		bk := beta[k]

		for idx := range T {
			dt := t - times[idx]

			if dt <= 0 {
				break
			}

			sum += stdmath.Exp(-bk * dt)
		}

		out[k] = mu[k] + alpha[k]*sum
	}
}
