//go:build !amd64 && !arm64

package activation

var (
	leakyReLUSlopeF32Funcs = []paramSlopeKernelImpl{{LeakyReLUSlopeF32Generic, "generic", true}}
	preluF32Funcs          = []paramSlopeKernelImpl{{PReLUF32Generic, "generic", true}}
	thresholdF32Funcs      = []paramSlopeKernelImpl{{ThresholdF32Generic, "generic", true}}
	hardTanhRangeF32Funcs  = []paramRangeKernelImpl{{HardTanhRangeF32Generic, "generic", true}}
	eluAlphaF32Funcs       = []paramSlopeKernelImpl{{ELUAlphaF32Generic, "generic", true}}
	celuAlphaF32Funcs      = []paramSlopeKernelImpl{{CELUAlphaF32Generic, "generic", true}}
	hardShrinkF32Funcs     = []paramSlopeKernelImpl{{HardShrinkF32Generic, "generic", true}}
	softShrinkF32Funcs     = []paramSlopeKernelImpl{{SoftShrinkF32Generic, "generic", true}}
	snakeF32Funcs          = []paramSlopeKernelImpl{{SnakeF32Generic, "generic", true}}
	snakeParametricF32Funcs = []paramRangeKernelImpl{{SnakeParametricF32Generic, "generic", true}}
	rreluF32Funcs          = []paramRReluKernelImpl{{RReLUF32Generic, "generic", true}}
	preluVF32Funcs         = []paramIndexedKernelImpl{{PReLUVF32Generic, "generic", true}}
)
