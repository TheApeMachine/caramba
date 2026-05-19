package losses

type f32PairSumKernelImpl struct {
	kernel    func(predictions, targets *float32, count int) float32
	name      string
	available bool
}

func pickF32PairSumKernel(
	candidates []f32PairSumKernelImpl,
) func(predictions, targets *float32, count int) float32 {
	for _, candidate := range candidates {
		if candidate.available {
			return candidate.kernel
		}
	}

	panic("losses: no float32 pair-sum kernel available")
}
