package causal

import (
	"fmt"
	"math"
)

/*
DAGMarkovFactorization computes the joint log probability of observations
under the Markov factorization of a DAG with Gaussian conditionals:
log P(x) = Σ_i log P(x_i | x_{PA_i})

Each conditional P(x_i | x_{PA_i}) is Gaussian with:
  - mean = β_i^T x_{PA_i}  (estimated via OLS from data)
  - variance = residual variance of node i

shape = [N, T]
data[0] = X [T*N] — observation matrix, row i is observation t
data[1] = adj [N*N] — DAG adjacency matrix; adj[i,j]=1 means j is parent of i
Returns log_prob [T] — log probability of each observation under the DAG Markov factorization.
*/
type DAGMarkovFactorization struct{}

/*
NewDAGMarkovFactorization instantiates a DAGMarkovFactorization operation.
It computes the Markov-factorized log probability under a DAG structure assumption.
*/
func NewDAGMarkovFactorization() *DAGMarkovFactorization {
	return &DAGMarkovFactorization{}
}

/*
Forward computes per-observation log probabilities under the DAG Markov factorization.
*/
func (dagMarkov *DAGMarkovFactorization) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Errorf("causal: DAGMarkovFactorization.Forward: len(shape)=%d, need >= 2", len(shape)).Error())
	}

	if len(data) < 2 {
		panic(fmt.Errorf("causal: DAGMarkovFactorization.Forward: len(data)=%d, need >= 2", len(data)).Error())
	}

	n := shape[0]
	t := shape[1]

	xMat := data[0]
	adj := data[1]

	if len(xMat) != t*n {
		panic(fmt.Errorf(
			"causal: DAGMarkovFactorization.Forward: len(X)=%d, need T*N=%d",
			len(xMat), t*n,
		).Error())
	}

	if len(adj) != n*n {
		panic(fmt.Errorf(
			"causal: DAGMarkovFactorization.Forward: len(adj)=%d, need N*N=%d",
			len(adj), n*n,
		).Error())
	}

	validateDAGAcyclic(adj, n)

	// For each node, identify parents and fit a Gaussian conditional model.
	// beta[i] [numParents] and sigma2[i] are learned from data.
	nodeBetas := make([][]float64, n)
	nodeSigma2 := make([]float64, n)
	nodeParents := make([][]int, n)

	for nodeIdx := 0; nodeIdx < n; nodeIdx++ {
		parents := make([]int, 0, n)

		for j := 0; j < n; j++ {
			if adj[nodeIdx*n+j] != 0 {
				parents = append(parents, j)
			}
		}

		nodeParents[nodeIdx] = parents
		np := len(parents)

		// Extract node values [T].
		nodeVals := make([]float64, t)

		for obsIdx := 0; obsIdx < t; obsIdx++ {
			nodeVals[obsIdx] = xMat[obsIdx*n+nodeIdx]
		}

		if np == 0 {
			// Root node: Gaussian with sample mean and variance.
			mean := 0.0

			for _, v := range nodeVals {
				mean += v
			}

			mean /= float64(t)

			variance := 0.0

			for _, v := range nodeVals {
				diff := v - mean
				variance += diff * diff
			}

			variance /= float64(t)

			if variance < 1e-10 {
				variance = 1e-10
			}

			nodeBetas[nodeIdx] = []float64{mean}
			nodeSigma2[nodeIdx] = variance
		} else if t <= np {
			nodeBetas[nodeIdx] = make([]float64, np+1)
			nodeSigma2[nodeIdx] = 1e-10
		} else {
			// Node with parents: fit Y ~ X_parents via OLS.
			parentMat := make([]float64, t*np)

			for obsIdx := 0; obsIdx < t; obsIdx++ {
				for pIdx, parentNode := range parents {
					parentMat[obsIdx*np+pIdx] = xMat[obsIdx*n+parentNode]
				}
			}

			beta := fitOLS(parentMat, nodeVals, makeRange(t), np)
			nodeBetas[nodeIdx] = beta

			// Compute residual variance.
			residualSS := 0.0

			for obsIdx := 0; obsIdx < t; obsIdx++ {
				parentRow := parentMat[obsIdx*np : (obsIdx+1)*np]
				predicted := beta[0] + applyDotProduct(beta[1:], parentRow)
				residual := nodeVals[obsIdx] - predicted
				residualSS += residual * residual
			}

			sigma2 := residualSS / float64(t)

			if sigma2 < 1e-10 {
				sigma2 = 1e-10
			}

			nodeSigma2[nodeIdx] = sigma2
		}
	}

	// Compute log P(x^t) = Σ_i log P(x_i^t | x_{PA_i}^t) for each observation.
	logProb := make([]float64, t)

	for obsIdx := 0; obsIdx < t; obsIdx++ {
		logP := 0.0

		for nodeIdx := 0; nodeIdx < n; nodeIdx++ {
			parents := nodeParents[nodeIdx]
			np := len(parents)
			sigma2 := nodeSigma2[nodeIdx]
			xVal := xMat[obsIdx*n+nodeIdx]

			var predicted float64

			if np == 0 {
				// Root node: mean is stored in beta[0].
				predicted = nodeBetas[nodeIdx][0]
			} else {
				parentRow := make([]float64, np)

				for pIdx, parentNode := range parents {
					parentRow[pIdx] = xMat[obsIdx*n+parentNode]
				}

				predicted = nodeBetas[nodeIdx][0] + applyDotProduct(nodeBetas[nodeIdx][1:], parentRow)
			}

			// log N(xVal; predicted, sigma2) = -0.5*log(2*pi*sigma2) - 0.5*(xVal-predicted)^2/sigma2
			diff := xVal - predicted
			logP += -0.5*math.Log(2*math.Pi*sigma2) - 0.5*diff*diff/sigma2
		}

		logProb[obsIdx] = logP
	}

	return logProb
}

/*
makeRange returns a slice [0, 1, ..., n-1].
*/
func makeRange(n int) []int {
	indices := make([]int, n)

	for idx := range indices {
		indices[idx] = idx
	}

	return indices
}

/*
validateDAGAcyclic panics if adj encodes a directed cycle. Entry adj[i*n+j] != 0 means
edge j → i (j is a parent of i), matching node parent enumeration in Forward.
*/
func validateDAGAcyclic(adj []float64, n int) {
	indeg := make([]int, n)

	for child := 0; child < n; child++ {
		for parent := 0; parent < n; parent++ {
			if adj[child*n+parent] != 0 {
				indeg[child]++
			}
		}
	}

	queue := make([]int, 0, n)

	for nodeIdx := 0; nodeIdx < n; nodeIdx++ {
		if indeg[nodeIdx] == 0 {
			queue = append(queue, nodeIdx)
		}
	}

	ordered := 0

	for head := 0; head < len(queue); head++ {
		nodeIdx := queue[head]
		ordered++

		for child := 0; child < n; child++ {
			if adj[child*n+nodeIdx] == 0 {
				continue
			}

			indeg[child]--

			if indeg[child] == 0 {
				queue = append(queue, child)
			}
		}
	}

	if ordered != n {
		panic(fmt.Errorf("causal: DAGMarkovFactorization.Forward: adjacency is not a DAG (cycle detected)"))
	}
}
