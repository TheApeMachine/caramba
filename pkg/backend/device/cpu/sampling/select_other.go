//go:build !amd64

package sampling

func GreedySampleFloat32Native(logits []float32) int32 {
	if len(logits) == 0 {
		return 0
	}

	return greedySampleF32Kernel(&logits[0], len(logits))
}

func SamplingSoftmaxRowFloat32Native(logits, out []float32, temperature float32) {
	if len(logits) == 0 {
		return
	}

	samplingSoftmaxRowF32Kernel(&logits[0], &out[0], temperature, len(logits))
}

var greedySampleF32Funcs = []f32GreedyKernelImpl{
	{greedySampleF32Generic, "generic", true},
}

var samplingSoftmaxRowF32Funcs = []f32SoftmaxRowKernelImpl{
	{samplingSoftmaxRowF32Generic, "generic", true},
}
