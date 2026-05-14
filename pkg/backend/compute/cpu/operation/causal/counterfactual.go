package causal

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Counterfactual runs Pearl's abduction–action–prediction for a heterogeneous linear SCM:

	Y_i = beta_i * X_i + epsilon_i

with one structural slope beta_i per observation (data[2]), not a shared global beta.

  - Abduction: epsilon_i = Y_obs[i] - beta_i * X_obs[i] per row.
  - Action: intervene to set X to each counterfactual level x' in X_cf.
  - Prediction: Y_cf[i,j] = beta_i * X_cf[j] + epsilon_i

shape = [N, N_cf]
data[0] = X_obs [N]
data[1] = Y_obs [N]
data[2] = beta [N]  — per-observation slopes
data[3] = X_cf [N_cf]
Returns Y_cf as a flat row-major [N × N_cf] slice of length N*N_cf (index i*N_cf + j).
*/
type Counterfactual struct{}

/*
NewCounterfactual instantiates a Counterfactual operation.
It implements Pearl's three-step abduction-action-prediction procedure.
*/
func NewCounterfactual() *Counterfactual {
	return &Counterfactual{}
}

/*
Forward runs the counterfactual query.
*/
func (counterfactual *Counterfactual) Forward(
	stateDict *state.Dict,
) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 2 {
		return nil, fmt.Errorf("causal.counterfactual: len(shape)=%d, need >= 2", len(shape))
	}

	if err := stateDict.RequireOperationInputs("causal.counterfactual", 4); err != nil {
		return nil, err
	}

	n := shape[0]
	nCF := shape[1]

	if n <= 0 {
		return nil, fmt.Errorf("causal.counterfactual: N=%d from shape[0] must be positive", n)
	}

	if nCF < 0 {
		return nil, fmt.Errorf("causal.counterfactual: N_cf=%d from shape[1] must be non-negative", nCF)
	}

	xObs := stateDict.Inputs[0]
	yObs := stateDict.Inputs[1]
	beta := stateDict.Inputs[2]
	xCF := stateDict.Inputs[3]

	if len(xObs) != n || len(yObs) != n || len(beta) != n {
		return nil, fmt.Errorf("causal.counterfactual: X_obs/Y_obs/beta must have len N=%d", n)
	}

	if len(xCF) != nCF {
		return nil, fmt.Errorf(
			"causal.counterfactual: len(X_cf)=%d, need N_cf=%d",
			len(xCF), nCF,
		)
	}

	epsilon := make([]float64, n)

	for obsIdx := 0; obsIdx < n; obsIdx++ {
		epsilon[obsIdx] = yObs[obsIdx] - beta[obsIdx]*xObs[obsIdx]
	}

	stateDict.EnsureOperationOutLen(n * nCF)

	for obsIdx := 0; obsIdx < n; obsIdx++ {
		base := obsIdx * nCF

		for cfIdx := 0; cfIdx < nCF; cfIdx++ {
			stateDict.Out[base+cfIdx] = beta[obsIdx]*xCF[cfIdx] + epsilon[obsIdx]
		}
	}

	return stateDict, nil
}
