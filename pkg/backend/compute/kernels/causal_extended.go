package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Additional causal-inference kernels: counterfactual, instrumental-
variable estimator, DAG-Markov factorization. These cover the
remaining causal package surface from the original substrate.
*/

func init() {
	Default.Register(Kernel{
		Name: "counterfactual",
		Signature: Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32, dtype.Float32, dtype.Float32, dtype.Float32,
			},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runCounterfactual,
	})

	Default.Register(Kernel{
		Name: "iv_estimate",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runIVEstimate,
	})

	Default.Register(Kernel{
		Name: "dag_markov_factorization",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Int32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runDAGMarkovFactorization,
	})

	Default.Register(Kernel{
		Name: "markov_flow_active",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Int32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMarkovFlowActive,
	})

	Default.Register(Kernel{
		Name: "markov_flow_internal",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Int32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMarkovFlowInternal,
	})
}

/*
runCounterfactual computes Y_x'(u) given observed Y(u), X(u) under
factual world and the counterfactual X = x'. The simplest twin-world
estimator uses Y_x'(u) = Y(u) + slope × (x' - X(u)), where slope is
the local treatment effect.

Args: (observedY, observedX, counterfactualX, slope, output).
*/
func runCounterfactual(args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	observedY, _ := args[0].Float32Native()
	observedX, _ := args[1].Float32Native()
	counterfactualX, _ := args[2].Float32Native()
	slope, _ := args[3].Float32Native()
	out, _ := args[4].Float32Native()

	if len(observedY) != len(observedX) || len(observedY) != len(counterfactualX) ||
		len(out) != len(observedY) || len(slope) < 1 {
		return tensor.ErrShapeMismatch
	}

	slopeValue := slope[0]

	counterfactualFloat32Native(out, observedY, observedX, counterfactualX, slopeValue)

	return nil
}

/*
runIVEstimate is the standard two-stage least-squares instrumental
variable estimator: β = Cov(Z, Y) / Cov(Z, X). Args:
(instrument Z, treatment X, outcome Y, output_scalar β).
*/
func runIVEstimate(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	instrument, _ := args[0].Float32Native()
	treatment, _ := args[1].Float32Native()
	outcome, _ := args[2].Float32Native()
	out, _ := args[3].Float32Native()

	n := len(instrument)

	if len(treatment) != n || len(outcome) != n || len(out) < 1 || n < 2 {
		return tensor.ErrShapeMismatch
	}

	out[0] = ivEstimateFloat32Native(instrument, treatment, outcome)

	return nil
}

/*
runDAGMarkovFactorization computes the joint P(X_1, ..., X_n) under
a Bayesian-network factorization given conditional probabilities and
a topological order. Args: (conditional [N, max_parents+1], parents_index [N, max_parents],
output_scalar joint).

Simplified: we accept a CPD matrix where row i carries the
conditional P(X_i = value | parents) and parents_index lists the
parents per variable. The output is the product of all per-variable
conditionals at a particular assignment encoded inline.
*/
func runDAGMarkovFactorization(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	conditionals, _ := args[0].Float32Native()
	parents, _ := args[1].Int32Native()
	out, _ := args[2].Float32Native()

	if len(out) < 1 || len(conditionals) == 0 {
		return tensor.ErrShapeMismatch
	}

	_ = parents

	// Reference factorization: product of per-variable
	// conditionals. The caller supplies the values already
	// indexed (one float32 per variable). Real DAG inference goes
	// through the orchestrator's structured plan; this kernel
	// just multiplies the supplied conditional probabilities.
	product := float64(1)

	for _, conditional := range conditionals {
		product *= math.Max(1e-12, float64(conditional))
	}

	out[0] = float32(product)
	return nil
}

/*
runMarkovFlowActive computes the flow of information from internal
nodes to active nodes through the boundary of a Markov blanket.
Args: (mutual_information_matrix [N, N], partition_labels [N],
output_flow [N]).

Returns per-active-node flow magnitude as the sum of MI with
internal nodes.
*/
func runMarkovFlowActive(args ...tensor.Tensor) error {
	return markovFlowDirection(args, 2 /* active */)
}

/*
runMarkovFlowInternal mirrors runMarkovFlowActive for the internal
side of the boundary.
*/
func runMarkovFlowInternal(args ...tensor.Tensor) error {
	return markovFlowDirection(args, 0 /* internal */)
}

func markovFlowDirection(args []tensor.Tensor, targetLabel int32) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	mi, _ := args[0].Float32Native()
	partition, _ := args[1].Int32Native()
	out, _ := args[2].Float32Native()

	dims := args[0].Shape().Dims()

	if len(dims) != 2 || dims[0] != dims[1] ||
		len(partition) != dims[0] || len(out) != dims[0] {
		return tensor.ErrShapeMismatch
	}

	n := dims[0]

	markovFlowFloat32Native(mi, partition, out, n, targetLabel)

	return nil
}
