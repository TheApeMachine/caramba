//go:build !386 && !amd64 && !arm64

package losses

func MseSumFloat32Native(predictions, targets []float32) float32 {
	if len(predictions) == 0 {
		return 0
	}

	return mseSumF32Kernel(&predictions[0], &targets[0], len(predictions))
}

func MaeSumFloat32Native(predictions, targets []float32) float32 {
	if len(predictions) == 0 {
		return 0
	}

	return maeSumF32Kernel(&predictions[0], &targets[0], len(predictions))
}

var mseSumF32Funcs = []f32PairSumKernelImpl{
	{MseSumF32Generic, "generic", true},
}

var maeSumF32Funcs = []f32PairSumKernelImpl{
	{MaeSumF32Generic, "generic", true},
}
