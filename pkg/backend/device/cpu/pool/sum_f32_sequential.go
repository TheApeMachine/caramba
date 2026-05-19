package pool

/*
sumFloat32Sequential accumulates values in strict index order with float32
arithmetic. Adaptive average pooling uses this so NEON sum reduction (f64
accumulation) does not diverge from the scalar reference.
*/
func sumFloat32Sequential(values []float32) float32 {
	var sum float32

	for _, value := range values {
		sum += value
	}

	return sum
}
