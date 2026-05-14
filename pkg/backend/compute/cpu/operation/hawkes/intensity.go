package hawkes

import (
	"fmt"
	"math"
	"sort"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
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

func (intensity *Intensity) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 2 {
		return nil, fmt.Errorf("hawkes.intensity: len(shape)=%d, need >= 2", len(shape))
	}

	if err := stateDict.RequireOperationInputs("hawkes.intensity", 5); err != nil {
		return nil, err
	}

	processCount := shape[0]
	eventCount := shape[1]
	times := stateDict.Inputs[0]
	alpha := stateDict.Inputs[1]
	beta := stateDict.Inputs[2]
	mu := stateDict.Inputs[3]
	timeSlice := stateDict.Inputs[4]

	if processCount <= 0 || eventCount < 0 {
		return nil, fmt.Errorf(
			"hawkes.intensity: need K > 0 and T >= 0 (got K=%d T=%d)",
			processCount, eventCount,
		)
	}

	if len(times) != eventCount {
		return nil, fmt.Errorf("hawkes.intensity: len(times)=%d, need T=%d", len(times), eventCount)
	}

	if len(alpha) != processCount || len(beta) != processCount || len(mu) != processCount {
		return nil, fmt.Errorf(
			"hawkes.intensity: alpha/beta/mu lengths must equal K=%d",
			processCount,
		)
	}

	if len(timeSlice) < 1 {
		return nil, fmt.Errorf("hawkes.intensity: current time input must have length >= 1")
	}

	currentTime := timeSlice[0]
	stateDict.EnsureOperationOutLen(processCount)
	applyIntensity(
		stateDict.Out,
		times,
		alpha,
		beta,
		mu,
		currentTime,
		processCount,
		eventCount,
	)

	return stateDict, nil
}

func applyIntensityScalar(out, times, alpha, beta, mu []float64, t float64, K, T int) {
	for k := 0; k < K; k++ {
		sum := 0.0
		bk := beta[k]

		cutoff := sort.Search(T, func(idx int) bool {
			return times[idx] >= t
		})

		for idx := 0; idx < cutoff; idx++ {
			sum += math.Exp(-bk * (t - times[idx]))
		}

		out[k] = mu[k] + alpha[k]*sum
	}
}
