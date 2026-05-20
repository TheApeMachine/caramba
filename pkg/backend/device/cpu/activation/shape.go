package activation

import "github.com/theapemachine/manifesto/tensor"

/*
PackedGatedShape derives batch and halfCount for packed gate+up layout.
The last dimension of dims is 2*halfCount; leading dimensions are batch axes.
*/
func PackedGatedShape(shape tensor.Shape) (batch, halfCount int, ok bool) {
	if !shape.Valid() {
		return 0, 0, false
	}

	dims := shape.Dims()

	if len(dims) == 0 {
		elements := shape.Len()

		if elements%2 != 0 {
			return 0, 0, false
		}

		return 1, elements / 2, true
	}

	halfCount = dims[len(dims)-1] / 2

	if dims[len(dims)-1]%2 != 0 {
		return 0, 0, false
	}

	batch = shape.Len() / (2 * halfCount)

	return batch, halfCount, true
}
