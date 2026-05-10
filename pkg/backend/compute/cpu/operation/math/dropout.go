package math

import (
	"math/rand"
)

// Dropout randomly zeros elements during training with probability P,
// scaling survivors by 1/(1-P). During inference it is an identity.
type Dropout struct {
	P        float64 // drop probability
	Training bool
}

func NewDropout(p float64, training bool) *Dropout {
	return &Dropout{P: p, Training: training}
}

func (op *Dropout) Forward(shape []int, data ...[]float64) []float64 {
	x := data[0]
	out := make([]float64, len(x))
	if !op.Training || op.P == 0 {
		copy(out, x)
		return out
	}
	scale := 1.0 / (1.0 - op.P)
	for i, v := range x {
		if rand.Float64() >= op.P {
			out[i] = v * scale
		}
		// else out[i] = 0 (zero value)
	}
	return out
}
