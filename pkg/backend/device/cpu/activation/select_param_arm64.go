//go:build arm64

package activation

func LeakyReLUSlopeF32NEON(dst, src *float32, count int, negativeSlope float32)
func PReLUF32NEON(dst, src *float32, count int, negativeSlope float32)
func ThresholdF32NEON(dst, src *float32, count int, threshold float32)
func HardTanhRangeF32NEON(dst, src *float32, count int, minVal, maxVal float32)

var (
	leakyReLUSlopeF32Funcs = []paramSlopeKernelImpl{
		{LeakyReLUSlopeF32NEON, "neon", true},
		{LeakyReLUSlopeF32Generic, "generic", true},
	}
	preluF32Funcs = []paramSlopeKernelImpl{
		{PReLUF32NEON, "neon", true},
		{PReLUF32Generic, "generic", true},
	}
	thresholdF32Funcs = []paramSlopeKernelImpl{
		{ThresholdF32NEON, "neon", true},
		{ThresholdF32Generic, "generic", true},
	}
	hardTanhRangeF32Funcs = []paramRangeKernelImpl{
		{HardTanhRangeF32NEON, "neon", true},
		{HardTanhRangeF32Generic, "generic", true},
	}
	eluAlphaF32Funcs = []paramSlopeKernelImpl{
		{ELUAlphaF32NEON, "neon", true},
		{ELUAlphaF32Generic, "generic", true},
	}
	celuAlphaF32Funcs = []paramSlopeKernelImpl{
		{CELUAlphaF32NEON, "neon", true},
		{CELUAlphaF32Generic, "generic", true},
	}
	hardShrinkF32Funcs = []paramSlopeKernelImpl{
		{HardShrinkF32NEON, "neon", true},
		{HardShrinkF32Generic, "generic", true},
	}
	softShrinkF32Funcs = []paramSlopeKernelImpl{
		{SoftShrinkF32NEON, "neon", true},
		{SoftShrinkF32Generic, "generic", true},
	}
	snakeF32Funcs = []paramSlopeKernelImpl{
		{SnakeF32NEON, "neon", true},
		{SnakeF32Generic, "generic", true},
	}
	snakeParametricF32Funcs = []paramRangeKernelImpl{
		{SnakeParametricF32NEON, "neon", true},
		{SnakeParametricF32Generic, "generic", true},
	}
	rreluF32Funcs = []paramRReluKernelImpl{
		{RReLUF32NEON, "neon", true},
		{RReLUF32Generic, "generic", true},
	}
	preluVF32Funcs = []paramIndexedKernelImpl{
		{PReLUVF32NEON, "neon", true},
		{PReLUVF32Generic, "generic", true},
	}
)

func ELUAlphaF32NEON(dst, src *float32, count int, alpha float32)
func CELUAlphaF32NEON(dst, src *float32, count int, alpha float32)
func HardShrinkF32NEON(dst, src *float32, count int, lambda float32)
func SoftShrinkF32NEON(dst, src *float32, count int, lambda float32)
func SnakeF32NEON(dst, src *float32, count int, alpha float32)
func SnakeParametricF32NEON(dst, src *float32, count int, alpha, beta float32)
func RReLUF32NEON(dst, src *float32, count int, lower, upper float32)
func PReLUVF32NEON(dst, src, slopes *float32, count int)
