package causal

import "fmt"

/*
Counterfactual computes E[Y_{X=x'}|X=x, Y=y] using the three-step
abduction-action-prediction procedure for linear SCMs.

Given a linear SCM: Y = beta * X + e
  - Abduction: infer noise e = Y - beta * X from observed (X=x, Y=y)
  - Action: modify SCM to fix X = x'
  - Prediction: compute Y' = beta * x' + e under counterfactual X

shape = [N, N_cf]
data[0] = X_obs [N]      — observed treatment values
data[1] = Y_obs [N]      — observed outcome values
data[2] = beta [N]       — linear causal coefficients (one per observation)
data[3] = X_cf [N_cf]    — counterfactual treatment values
Returns Y_cf [N_cf] — predicted counterfactual outcomes.
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
func (counterfactual *Counterfactual) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Errorf("causal: Counterfactual.Forward: len(shape)=%d, need >= 2", len(shape)).Error())
	}

	if len(data) < 4 {
		panic(fmt.Errorf("causal: Counterfactual.Forward: len(data)=%d, need >= 4", len(data)).Error())
	}

	n := shape[0]
	nCF := shape[1]

	xObs := data[0]
	yObs := data[1]
	beta := data[2]
	xCF := data[3]

	if len(xObs) != n || len(yObs) != n || len(beta) != n {
		panic(fmt.Errorf(
			"causal: Counterfactual.Forward: X_obs/Y_obs/beta must have len N=%d", n,
		).Error())
	}

	if len(xCF) != nCF {
		panic(fmt.Errorf(
			"causal: Counterfactual.Forward: len(X_cf)=%d, need N_cf=%d",
			len(xCF), nCF,
		).Error())
	}

	// Step 1 — Abduction: compute mean exogenous noise from observed data.
	// e_i = y_i - beta_i * x_i
	// Average noise represents the exogenous background factor.
	noiseSum := 0.0

	for obsIdx := 0; obsIdx < n; obsIdx++ {
		noiseSum += yObs[obsIdx] - beta[obsIdx]*xObs[obsIdx]
	}

	meanNoise := noiseSum / float64(n)

	// Mean beta (structural coefficient under counterfactual).
	betaSum := 0.0

	for obsIdx := 0; obsIdx < n; obsIdx++ {
		betaSum += beta[obsIdx]
	}

	meanBeta := betaSum / float64(n)

	// Step 2 — Action: fix X = x' (graph surgery handled implicitly).
	// Step 3 — Prediction: Y' = meanBeta * x' + meanNoise
	yCF := make([]float64, nCF)

	for cfIdx := 0; cfIdx < nCF; cfIdx++ {
		yCF[cfIdx] = meanBeta*xCF[cfIdx] + meanNoise
	}

	return yCF
}
