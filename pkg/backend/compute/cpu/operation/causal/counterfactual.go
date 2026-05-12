package causal

import "fmt"

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
func (_ *Counterfactual) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Errorf("causal: Counterfactual.Forward: len(shape)=%d, need >= 2", len(shape)))
	}

	if len(data) < 4 {
		panic(fmt.Errorf("causal: Counterfactual.Forward: len(data)=%d, need >= 4", len(data)))
	}

	n := shape[0]
	nCF := shape[1]

	if n <= 0 {
		panic(fmt.Errorf("causal: Counterfactual.Forward: N=%d from shape[0] must be positive", n))
	}

	xObs := data[0]
	yObs := data[1]
	beta := data[2]
	xCF := data[3]

	if len(xObs) != n || len(yObs) != n || len(beta) != n {
		panic(fmt.Errorf(
			"causal: Counterfactual.Forward: X_obs/Y_obs/beta must have len N=%d", n,
		))
	}

	if len(xCF) != nCF {
		panic(fmt.Errorf(
			"causal: Counterfactual.Forward: len(X_cf)=%d, need N_cf=%d",
			len(xCF), nCF,
		))
	}

	epsilon := make([]float64, n)

	for obsIdx := 0; obsIdx < n; obsIdx++ {
		epsilon[obsIdx] = yObs[obsIdx] - beta[obsIdx]*xObs[obsIdx]
	}

	yCF := make([]float64, n*nCF)

	for obsIdx := 0; obsIdx < n; obsIdx++ {
		base := obsIdx * nCF

		for cfIdx := 0; cfIdx < nCF; cfIdx++ {
			yCF[base+cfIdx] = beta[obsIdx]*xCF[cfIdx] + epsilon[obsIdx]
		}
	}

	return yCF
}
