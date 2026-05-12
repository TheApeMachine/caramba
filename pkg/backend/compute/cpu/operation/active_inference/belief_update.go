package active_inference

import "fmt"

/*
BeliefUpdate performs a gradient descent step on the variational free energy
to update Gaussian belief parameters mu and log_sigma.

Gradients of F w.r.t. parameters:

	dF/dmu        = mu + prediction_error
	dF/dlog_sigma = exp(log_sigma) - 1   (from the KL term)

Update rules:

	mu_new        = mu        - lr * (mu + prediction_error)
	log_sigma_new = log_sigma - lr * (exp(log_sigma) - 1)

shape must have len(shape) >= 2: shape[0] = N (batch / belief dimension), shape[1] = lr
encoded as an integer step count; the effective learning rate is

	lr := float64(shape[1]) * 1e-4

with 0 < lr <= 1 (i.e. 1 <= shape[1] <= 10000).

data[0] = mu [N], data[1] = log_sigma [N], data[2] = prediction_error [N]

Forward returns a vector of length 2*N: indices [0:N) are updated mu, [N:2*N) are updated log_sigma.
*/
type BeliefUpdate struct{}

/*
NewBeliefUpdate instantiates a new BeliefUpdate operation.
*/
func NewBeliefUpdate() *BeliefUpdate { return &BeliefUpdate{} }

/*
Forward computes one gradient descent step updating mu and log_sigma.
Requires shape[0] = N > 0 and shape[1] so lr = float64(shape[1])*1e-4 satisfies 0 < lr <= 1.
Returns [mu_new || log_sigma_new] of length 2*N.
*/
func (op *BeliefUpdate) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Sprintf("active_inference: BeliefUpdate.Forward: len(shape)=%d, need >= 2", len(shape)))
	}

	n := shape[0]

	if n <= 0 {
		panic(fmt.Sprintf("active_inference: BeliefUpdate.Forward: batch size n=%d from shape[0] must be positive", n))
	}

	lrStep := shape[1]
	lr := float64(lrStep) * 1e-4

	if lrStep <= 0 {
		panic(fmt.Sprintf(
			"active_inference: BeliefUpdate.Forward: shape[1]=%d yields non-positive lr=%g (need shape[1] >= 1)",
			lrStep, lr,
		))
	}

	if lr > 1.0 {
		panic(fmt.Sprintf(
			"active_inference: BeliefUpdate.Forward: shape[1]=%d yields lr=%g > 1 (cap at shape[1]=10000)",
			lrStep, lr,
		))
	}

	if len(data) < 3 {
		panic(fmt.Sprintf("active_inference: BeliefUpdate.Forward: len(data)=%d, need >= 3", len(data)))
	}

	if len(data[0]) != n {
		panic(fmt.Sprintf("active_inference: BeliefUpdate.Forward: len(mu)=%d, need N=%d", len(data[0]), n))
	}

	if len(data[1]) != n {
		panic(fmt.Sprintf("active_inference: BeliefUpdate.Forward: len(log_sigma)=%d, need N=%d", len(data[1]), n))
	}

	if len(data[2]) != n {
		panic(fmt.Sprintf("active_inference: BeliefUpdate.Forward: len(pred_err)=%d, need N=%d", len(data[2]), n))
	}

	out := make([]float64, 2*n)
	applyBeliefUpdate(out[:n], out[n:], data[0], data[1], data[2], lr, n)

	return out
}
