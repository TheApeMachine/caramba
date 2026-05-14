package hawkes

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

const lambdaStarEpsilon = 1e-12

/*
Simulate generates event times for K Hawkes processes using Ogata's thinning
algorithm. The simulation runs for each process independently.

shape = [K, T_max_steps]
data[0] = mu    [K]
data[1] = alpha [K]
data[2] = beta  [K]
data[3] = T_max [1]   (end time)

Returns event times packed as K blocks of T_max_steps float64 each,
where unused entries are filled with sentinel -1.
*/
type Simulate struct{}

func NewSimulate() *Simulate { return &Simulate{} }

func (simulate *Simulate) Forward(stateDict *state.Dict) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 2 {
		return nil, fmt.Errorf("hawkes.simulate: len(shape)=%d, need >= 2", len(shape))
	}

	if err := stateDict.RequireOperationInputs("hawkes.simulate", 4); err != nil {
		return nil, err
	}

	processCount := shape[0]
	maxSteps := shape[1]
	mu := stateDict.Inputs[0]
	alpha := stateDict.Inputs[1]
	beta := stateDict.Inputs[2]
	tMaxSlice := stateDict.Inputs[3]

	if processCount <= 0 || maxSteps < 0 {
		return nil, fmt.Errorf(
			"hawkes.simulate: need K > 0 and maxSteps >= 0 (got K=%d maxSteps=%d)",
			processCount, maxSteps,
		)
	}

	if len(mu) != processCount || len(alpha) != processCount || len(beta) != processCount {
		return nil, fmt.Errorf("hawkes.simulate: mu/alpha/beta must have length K=%d", processCount)
	}

	for index := range processCount {
		if beta[index] <= 0 {
			return nil, fmt.Errorf("hawkes.simulate: beta[%d] must be > 0", index)
		}

		if alpha[index] < 0 {
			return nil, fmt.Errorf("hawkes.simulate: alpha[%d] must be >= 0", index)
		}

		if mu[index] <= 0 {
			return nil, fmt.Errorf("hawkes.simulate: mu[%d] must be > 0", index)
		}
	}

	if len(tMaxSlice) < 1 {
		return nil, fmt.Errorf("hawkes.simulate: T_max must have length >= 1")
	}

	tMax := tMaxSlice[0]
	stateDict.EnsureOperationOutLen(processCount * maxSteps)

	for index := range processCount * maxSteps {
		stateDict.Out[index] = -1
	}

	for processIndex := range processCount {
		events := ogataThinningSingle(
			mu[processIndex],
			alpha[processIndex],
			beta[processIndex],
			tMax,
			maxSteps,
		)

		for eventIndex, eventTime := range events {
			stateDict.Out[processIndex*maxSteps+eventIndex] = eventTime
		}
	}

	return stateDict, nil
}

// ogataThinningSingle runs Ogata's thinning for a single Hawkes process.
func ogataThinningSingle(mu, alpha, beta, tMax float64, maxSteps int) []float64 {
	events := make([]float64, 0, maxSteps)
	t := 0.0

	for t < tMax && len(events) < maxSteps {
		lambdaStar := mu + hawkesExcitation(events, t, beta, alpha)

		if lambdaStar < lambdaStarEpsilon {
			lambdaStar = lambdaStarEpsilon
		}

		u1 := rand.Float64()

		for u1 <= 0 {
			u1 = rand.Float64()
		}

		dt := -math.Log(u1) / lambdaStar
		t += dt

		if t >= tMax {
			break
		}

		lambdaT := mu + hawkesExcitation(events, t, beta, alpha)

		if lambdaT < 0 {
			lambdaT = 0
		}

		if lambdaT > lambdaStar {
			lambdaT = lambdaStar
		}

		u2 := rand.Float64()

		if u2 <= lambdaT/lambdaStar {
			events = append(events, t)
		}
	}

	return events
}
