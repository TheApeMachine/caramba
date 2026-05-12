package bench

import (
	mathops "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

/*
Accuracy accumulates per-sample argmax accuracy.
Inputs: data[0] = predicted logits/probs, data[1] = one-hot targets.
Output: running mean accuracy as single-element slice.
*/
type Accuracy struct {
	correct int
	total   int
}

func NewAccuracy() *Accuracy { return &Accuracy{} }

func (acc *Accuracy) Forward(_ []int, data ...[]float64) []float64 {
	acc.total++

	if argmax(data[0]) == argmax(data[1]) {
		acc.correct++
	}

	return []float64{float64(acc.correct) / float64(acc.total)}
}

/*
Perplexity accumulates per-sample cross-entropy and computes exp(mean CE).
Inputs: data[0] = predicted probabilities, data[1] = one-hot targets.
Output: running perplexity as single-element slice.
*/
type Perplexity struct {
	sumCE float64
	total int
}

func NewPerplexity() *Perplexity { return &Perplexity{} }

func (px *Perplexity) Forward(_ []int, data ...[]float64) []float64 {
	px.total++

	n := len(data[0])
	probs := make([]float64, n)
	copy(probs, data[0])
	mathops.AddScalarVec(probs, 1e-9)

	logp := make([]float64, n)
	mathops.LogVec(logp, probs)
	mathops.MulVec(logp, logp, data[1])
	ce := -mathops.ReduceSum(logp)

	px.sumCE += ce

	// Single scalar exp via the SIMD primitive (1-element slice).
	avg := []float64{px.sumCE / float64(px.total)}
	mathops.ExpVec(avg, avg)

	return avg
}

/*
F1 accumulates binary classification TP/FP/FN and emits macro F1.
Inputs: data[0] = predicted probabilities (threshold 0.5), data[1] = binary targets.
Output: running F1 as single-element slice.
*/
type F1 struct {
	tp, fp, fn float64
}

func NewF1() *F1 { return &F1{} }

func (f1 *F1) Forward(_ []int, data ...[]float64) []float64 {
	threshold := 0.5

	for idx := range data[0] {
		pred := data[0][idx] >= threshold
		actual := data[1][idx] >= threshold

		switch {
		case pred && actual:
			f1.tp++
		case pred && !actual:
			f1.fp++
		case !pred && actual:
			f1.fn++
		}
	}

	precision := f1.tp / (f1.tp + f1.fp + 1e-9)
	recall := f1.tp / (f1.tp + f1.fn + 1e-9)

	return []float64{2 * precision * recall / (precision + recall + 1e-9)}
}

func argmax(xs []float64) int {
	return argmaxImpl(xs)
}

