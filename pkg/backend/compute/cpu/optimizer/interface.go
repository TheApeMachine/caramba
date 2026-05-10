package optimizer

/*
Optimizer updates a parameter vector given its gradient.
State (moments, step count, etc.) is owned by the implementation.
Each Step call returns the updated parameter vector; the input slices are not mutated.
*/
type Optimizer interface {
	Step(params, grads []float64) []float64
}
