package operation

/*
Operation applies an elementwise transform to a float64 slice and returns a new slice.
Each call may allocate; callers should not assume buffer reuse.
*/
type Operation interface {
	Forward(shape []int, data ...[]float64) []float64
}
