//go:build arm64

package kernels

func mseSumFloat32Native(predictions, targets []float32) float32 {
	if len(predictions) == 0 {
		return 0
	}

	return mseSumNEONAsm(&predictions[0], &targets[0], len(predictions))
}

func maeSumFloat32Native(predictions, targets []float32) float32 {
	if len(predictions) == 0 {
		return 0
	}

	return maeSumNEONAsm(&predictions[0], &targets[0], len(predictions))
}

func l1NormFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return l1NormNEONAsm(&values[0], len(values))
}
