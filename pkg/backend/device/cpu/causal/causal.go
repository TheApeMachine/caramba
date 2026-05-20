package causal

import (
	"math"

	"github.com/theapemachine/manifesto/tensor"
)

/*
Causal-inference primitives.

  - cholesky:               L L^T = A for symmetric positive-definite A.
  - backdoor_adjustment:    P(Y|do(X)) via the backdoor formula.
  - frontdoor_adjustment:   P(Y|do(X)) via the frontdoor formula.
  - do_calculus_intervene:  zero out incoming edges to the intervened
                            variable in an adjacency matrix.
  - counterfactual:         Y under counterfactual X' given observed Y, X.
  - average_treatment_effect (CATE): E[Y(1) - Y(0) | X = x].
*/

/*
runCholesky computes L such that L L^T = A in place, lower-triangular.
Args: (input_matrix, output_L). Both must be [N, N].
*/
func runCholesky(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	input, _ := args[0].Float32Native()
	out, _ := args[1].Float32Native()

	dims := args[0].Shape().Dims()

	if len(dims) != 2 || dims[0] != dims[1] {
		return tensor.ErrShapeMismatch
	}

	n := dims[0]

	if len(out) != n*n {
		return tensor.ErrShapeMismatch
	}

	for index := range out {
		out[index] = 0
	}

	for rowIndex := 0; rowIndex < n; rowIndex++ {
		for colIndex := 0; colIndex <= rowIndex; colIndex++ {
			sum := float64(input[rowIndex*n+colIndex])

			for innerIndex := 0; innerIndex < colIndex; innerIndex++ {
				sum -= float64(out[rowIndex*n+innerIndex]) * float64(out[colIndex*n+innerIndex])
			}

			switch {
			case rowIndex == colIndex:
				if sum <= 0 {
					return tensor.ErrShapeMismatch
				}

				out[rowIndex*n+colIndex] = float32(math.Sqrt(sum))
			default:
				out[rowIndex*n+colIndex] = float32(sum / float64(out[colIndex*n+colIndex]))
			}
		}
	}

	return nil
}

/*
runBackdoorAdjustment computes Σ_z P(Y|X, Z=z) × P(Z=z) for each
value of X. Args: (conditional [X, Z, Y], marginalZ [Z], output [X, Y]).
*/
func runBackdoorAdjustment(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	conditional, _ := args[0].Float32Native()
	marginalZ, _ := args[1].Float32Native()
	out, _ := args[2].Float32Native()

	dims := args[0].Shape().Dims()

	if len(dims) != 3 || dims[1] != len(marginalZ) ||
		len(out) != dims[0]*dims[2] {
		return tensor.ErrShapeMismatch
	}

	xCount := dims[0]
	zCount := dims[1]
	yCount := dims[2]

	BackdoorAdjustmentFloat32Native(
		conditional, marginalZ, out,
		xCount, zCount, yCount,
	)

	return nil
}

/*
runFrontdoorAdjustment computes the frontdoor formula:
P(Y|do(X)) = Σ_m P(M|X) × Σ_x' P(Y|X=x', M) × P(X=x').
Args: (mediatorGivenX [X, M], outcomeGivenXM [X, M, Y],
marginalX [X], output [X, Y]).
*/
func runFrontdoorAdjustment(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	mediatorGivenX, _ := args[0].Float32Native()
	outcomeGivenXM, _ := args[1].Float32Native()
	marginalX, _ := args[2].Float32Native()
	out, _ := args[3].Float32Native()

	mDims := args[0].Shape().Dims()
	oDims := args[1].Shape().Dims()

	if len(mDims) != 2 || len(oDims) != 3 {
		return tensor.ErrShapeMismatch
	}

	xCount := mDims[0]
	mCount := mDims[1]
	yCount := oDims[2]

	if oDims[0] != xCount || oDims[1] != mCount ||
		len(marginalX) != xCount || len(out) != xCount*yCount {
		return tensor.ErrShapeMismatch
	}

	for index := range out {
		out[index] = 0
	}

	FrontdoorAdjustmentFloat32Native(
		mediatorGivenX, outcomeGivenXM, marginalX, out,
		xCount, mCount, yCount,
	)

	return nil
}

/*
runDoIntervene removes incoming edges to a set of intervened nodes
in an adjacency matrix. Args: (adjacency [N, N], intervenedNodes [k],
output [N, N]).
*/
func runDoIntervene(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	adjacency, _ := args[0].Float32Native()
	intervened, _ := args[1].Int32Native()
	out, _ := args[2].Float32Native()

	dims := args[0].Shape().Dims()

	if len(dims) != 2 || dims[0] != dims[1] || len(out) != len(adjacency) {
		return tensor.ErrShapeMismatch
	}

	DoInterveneFloat32Native(out, adjacency, intervened, dims[0])

	return nil
}

/*
runCATE computes the conditional average treatment effect for each
covariate row: τ(x) = E[Y(1) - Y(0) | X = x]. Args: (predictedTreated
[N, Y], predictedControl [N, Y], output [N, Y]).
*/
func runCATE(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	treated, _ := args[0].Float32Native()
	control, _ := args[1].Float32Native()
	out, _ := args[2].Float32Native()

	if len(treated) != len(control) || len(out) != len(treated) {
		return tensor.ErrShapeMismatch
	}

	CateFloat32Native(treated, control, out)

	return nil
}
