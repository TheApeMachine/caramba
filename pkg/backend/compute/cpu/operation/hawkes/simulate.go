package hawkes

import (
	"fmt"
	"math"
	"math/rand/v2"
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

func (op *Simulate) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Errorf("hawkes: Simulate: len(shape)=%d, need >= 2", len(shape)))
	}

	if len(data) < 4 {
		panic(fmt.Errorf("hawkes: Simulate: len(data)=%d, need 4", len(data)))
	}

	K, maxSteps := shape[0], shape[1]
	mu := data[0]
	alpha := data[1]
	beta := data[2]
	tMaxSlice := data[3]

	if len(mu) != K || len(alpha) != K || len(beta) != K {
		panic(fmt.Errorf("hawkes: Simulate: mu/alpha/beta must have length K=%d", K))
	}

	for idx := range K {
		if beta[idx] <= 0 {
			panic(fmt.Errorf("hawkes: Simulate: beta[%d] must be > 0", idx))
		}

		if alpha[idx] < 0 {
			panic(fmt.Errorf("hawkes: Simulate: alpha[%d] must be >= 0", idx))
		}

		if mu[idx] <= 0 {
			panic(fmt.Errorf("hawkes: Simulate: mu[%d] must be > 0", idx))
		}
	}

	if len(tMaxSlice) < 1 {
		panic(fmt.Errorf("hawkes: Simulate: data[3] (T_max) must have length >= 1"))
	}

	tMax := tMaxSlice[0]
	out := make([]float64, K*maxSteps)

	for idx := range K * maxSteps {
		out[idx] = -1
	}

	for k := range K {
		events := ogataThinningSingle(mu[k], alpha[k], beta[k], tMax, maxSteps)

		for idx, ev := range events {
			out[k*maxSteps+idx] = ev
		}
	}

	return out
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
