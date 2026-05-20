package interpretability

/*
ActivationSteerFloat32Native applies a steering vector to an activation buffer.
*/
func ActivationSteerFloat32Native(
	destination, base, direction []float32,
	coefficient float32,
) {
	elementCount := len(destination)

	if elementCount == 0 {
		return
	}

	activationSteerFloat32Kernel(destination, base, direction, coefficient, elementCount)
}
