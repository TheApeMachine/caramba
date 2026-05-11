package operation

/*
Operation applies an elementwise transform to a float64 slice and returns a new slice.
Each call may allocate; callers should not assume buffer reuse.
*/
type Operation interface {
	Forward(shape []int, data ...[]float64) []float64
}

/*
Parameterized is implemented by operations that own learnable parameters.
Weights/LoadWeights on the Graph use this to snapshot and restore model state.
*/
type Parameterized interface {
	Params() []float64
	SetParams([]float64)
}
