//go:build arm64

package cpu

func MseSumFloat32Native(predictions, targets []float32) float32 {
	if len(predictions) == 0 {
		return 0
	}

	return MseSumNEONAsm(&predictions[0], &targets[0], len(predictions))
}

func MaeSumFloat32Native(predictions, targets []float32) float32 {
	if len(predictions) == 0 {
		return 0
	}

	return MaeSumNEONAsm(&predictions[0], &targets[0], len(predictions))
}

func L1NormFloat32Native(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}

	return L1NormNEONAsm(&values[0], len(values))
}
