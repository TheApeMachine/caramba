package math

/*
Sign applies the elementwise sign function: out[i] = +1 if src[i] > 0, -1 if < 0, 0 if == 0.
Used as the activation in classic discrete Hopfield networks.
shape: [N].
*/
type Sign struct{}

func NewSign() *Sign { return &Sign{} }

func (op *Sign) Forward(shape []int, data ...[]float64) []float64 {
	if len(data) == 0 || len(data[0]) == 0 {
		return nil
	}
	out := make([]float64, len(data[0]))
	signVec(out, data[0])
	return out
}
