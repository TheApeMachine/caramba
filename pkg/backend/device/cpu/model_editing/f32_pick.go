package model_editing

type weightGraftAddKernelImpl struct {
	kernel    func(weights, injection []float32, count int)
	name      string
	available bool
}

func pickWeightGraftAddKernel(
	candidates []weightGraftAddKernelImpl,
) func(weights, injection []float32, count int) {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("model_editing: no weight graft add kernel available")
}
