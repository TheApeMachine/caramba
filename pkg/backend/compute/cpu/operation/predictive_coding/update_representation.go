package predictive_coding

import "fmt"

/*
UpdateRepresentation performs the representation update step of predictive coding:
r += lr * (W^T @ ε_lower - ε_self)
where W^T @ ε_lower is the bottom-up error signal propagated through transposed
generative weights, and ε_self is the prediction error from the layer above.
shape=[D_in, D_out, lr_bits], data[0]=r [D_in], data[1]=W [D_out*D_in],
data[2]=eps_lower [D_out], data[3]=eps_self [D_in], data[4]=lr [1] → r_new [D_in].
*/
type UpdateRepresentation struct{}

/*
NewUpdateRepresentation instantiates a new UpdateRepresentation operation.
*/
func NewUpdateRepresentation() *UpdateRepresentation { return &UpdateRepresentation{} }

/*
Forward computes r_new = r + lr * (W^T @ eps_lower - eps_self).
*/
func (op *UpdateRepresentation) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Sprintf("predictive_coding: UpdateRepresentation.Forward: len(shape)=%d, need >= 2", len(shape)))
	}

	dIn, dOut := shape[0], shape[1]

	if len(data) < 5 {
		panic(fmt.Sprintf("predictive_coding: UpdateRepresentation.Forward: len(data)=%d, need >= 5", len(data)))
	}

	r, W, epsLower, epsSelf, lrVec := data[0], data[1], data[2], data[3], data[4]

	if len(r) != dIn || len(W) != dOut*dIn || len(epsLower) != dOut || len(epsSelf) != dIn {
		panic(fmt.Sprintf(
			"predictive_coding: UpdateRepresentation.Forward: shape mismatch r=%d W=%d eps_lower=%d eps_self=%d",
			len(r), len(W), len(epsLower), len(epsSelf),
		))
	}

	lr := lrVec[0]
	signal := make([]float64, dIn)

	// W^T @ eps_lower: signal[i] = sum_j W[j*dIn+i] * eps_lower[j]
	applyMatVecTranspose(signal, W, epsLower, dOut, dIn)

	// signal -= eps_self
	applySubVecInPlace(signal, epsSelf)

	// r_new = r + lr * signal
	out := make([]float64, dIn)
	copy(out, r)
	applyAxpy(out, signal, lr)

	return out
}
