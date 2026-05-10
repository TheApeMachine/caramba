package math

import gomath "math"

// Log computes log(x) elementwise. Pure Go.
type Log struct{}

func NewLog() *Log { return &Log{} }

func (op *Log) Forward(shape []int, data ...[]float64) []float64 {
	x := data[0]
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = gomath.Log(v)
	}
	return out
}
