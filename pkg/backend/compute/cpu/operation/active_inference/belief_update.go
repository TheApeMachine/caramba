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

shape = [N, lr]:
  shape[0] = N   (belief dimension)
  shape[1] = lr  encoded as float bits — caller passes lr via shape[1] which is
             reinterpreted through math.Float64frombits; see callers for convention.

Actually lr is passed as a separate scalar via shape[1] cast to float64 in bits.
We adopt the convention: shape[1] is the raw bit pattern of lr as int.
For simplicity lr is passed as data[3] (single-element slice).

data[0] = mu [N], data[1] = log_sigma [N], data[2] = prediction_error [N]
shape   = [N]  (lr is shape[1] if len(shape)>=2, cast to float64)
→ [2N]: first N elements are updated mu, next N are updated log_sigma.
*/
type BeliefUpdate struct{}

/*
NewBeliefUpdate instantiates a new BeliefUpdate operation.
*/
func NewBeliefUpdate() *BeliefUpdate { return &BeliefUpdate{} }

/*
Forward computes one gradient descent step updating mu and log_sigma.
shape[0] = N, shape[1] = lr as integer bits (use math.Float64bits).
Returns [mu_new || log_sigma_new] of length 2N.
*/
func (op *BeliefUpdate) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 2 {
		panic(fmt.Sprintf("active_inference: BeliefUpdate.Forward: len(shape)=%d, need >= 2", len(shape)))
	}

	n := shape[0]

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

	lr := float64(shape[1]) * 1e-4 // shape[1] encodes lr as integer steps of 1e-4

	out := make([]float64, 2*n)
	applyBeliefUpdate(out[:n], out[n:], data[0], data[1], data[2], lr, n)

	return out
}
