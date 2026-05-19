package math

const QuickGeluScale = float32(1.702)

func FastHardGelu32(value float32) float32 {
	inner := value + 3

	if inner < 0 {
		inner = 0
	}

	if inner > 6 {
		inner = 6
	}

	return value * (inner / 6)
}

func FastHardGelu64(value float64) float64 {
	return float64(FastHardGelu32(float32(value)))
}

func FastQuickGelu32(value float32) float32 {
	return value * FastSigmoid32(QuickGeluScale*value)
}

func FastQuickGelu64(value float64) float64 {
	return float64(FastQuickGelu32(float32(value)))
}

func FastTanhShrink32(value float32) float32 {
	return value - FastTanh32(value)
}

func FastTanhShrink64(value float64) float64 {
	return float64(FastTanhShrink32(float32(value)))
}

func FastHardShrink32(value, lambda float32) float32 {
	if value > lambda || value < -lambda {
		return value
	}

	return 0
}

func FastHardShrink64(value float64, lambda float64) float64 {
	return float64(FastHardShrink32(float32(value), float32(lambda)))
}

func FastSoftShrink32(value, lambda float32) float32 {
	if value > lambda {
		return value - lambda
	}

	if value < -lambda {
		return value + lambda
	}

	return 0
}

func FastSoftShrink64(value float64, lambda float64) float64 {
	return float64(FastSoftShrink32(float32(value), float32(lambda)))
}

func FastRReLU32(value, slope float32) float32 {
	if value > 0 {
		return value
	}

	return value * slope
}

func FastSnakeParametric32(value, alpha, beta float32) float32 {
	sine := FastSin32(alpha * value)
	return value + (1/beta)*sine*sine
}

func FastSnakeParametric64(value, alpha, beta float64) float64 {
	return float64(FastSnakeParametric32(float32(value), float32(alpha), float32(beta)))
}

func FastLinGLU32(gate, up float32) float32 {
	return gate * up
}

func FastSeGLU32(gate, up float32) float32 {
	return up * FastSigmoid32(gate)
}

func FastLinGLU64(gate, up float64) float64 {
	return gate * up
}

func FastSeGLU64(gate, up float64) float64 {
	return float64(FastSeGLU32(float32(gate), float32(up)))
}
