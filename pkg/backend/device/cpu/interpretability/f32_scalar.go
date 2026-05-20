package interpretability

/*
ActivationSteerFloat32Scalar computes dst[i] = base[i] + coefficient * direction[i].
*/
func ActivationSteerFloat32Scalar(
	destination, base, direction []float32,
	coefficient float32,
) {
	for index := range destination {
		destination[index] = base[index] + coefficient*direction[index]
	}
}
