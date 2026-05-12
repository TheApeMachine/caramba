package math

// Log computes log(x) elementwise via vectorized SIMD assembly.
type Log struct{}

func NewLog() *Log { return &Log{} }

func (op *Log) Forward(shape []int, data ...[]float64) []float64 {
	x := data[0]
	out := make([]float64, len(x))
	logVec(out, x)

	return out
}
