package predictive_coding

import "fmt"

/*
UpdateWeights performs the Hebbian weight update step of predictive coding:
ΔW = lr * ε @ r^T, giving W_new = W + ΔW.
The generative weights are updated to reduce future prediction errors.
shape=[D_out, D_in], data[0]=W [D_out*D_in], data[1]=eps [D_out],
data[2]=r [D_in], data[3]=lr [1] → W_new [D_out*D_in].
*/
type UpdateWeights struct{}

/*
NewUpdateWeights instantiates a new UpdateWeights operation.
*/
func NewUpdateWeights() *UpdateWeights { return &UpdateWeights{} }

/*
Forward computes W_new[i*D_in+j] = W[i*D_in+j] + lr * eps[i] * r[j].
*/
func (op *UpdateWeights) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Sprintf("predictive_coding: UpdateWeights.Forward: len(shape)=%d, need >= 2", len(shape)))
	}

	dOut, dIn := shape[0], shape[1]

	if len(data) < 4 {
		panic(fmt.Sprintf("predictive_coding: UpdateWeights.Forward: len(data)=%d, need >= 4", len(data)))
	}

	W, eps, r, lrVec := data[0], data[1], data[2], data[3]

	if len(W) != dOut*dIn || len(eps) != dOut || len(r) != dIn {
		panic(fmt.Sprintf(
			"predictive_coding: UpdateWeights.Forward: shape mismatch W=%d eps=%d r=%d",
			len(W), len(eps), len(r),
		))
	}

	lr := lrVec[0]
	out := make([]float64, dOut*dIn)
	copy(out, W)

	// ΔW[i,j] = lr * eps[i] * r[j] — outer product scaled by lr
	applyOuterAdd(out, eps, r, lr, dOut, dIn)

	return out
}
