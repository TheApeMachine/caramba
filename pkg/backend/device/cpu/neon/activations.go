package neon

import "math"

/*
Activation kernels with the [N] in → [N] out signature. Each
registers under the canonical activation name. Variants with
hyperparameters (LeakyReLU's slope, ELU's alpha, etc.) take their
default values here; the orchestrator binds custom values via the
*Config helpers when needed.
*/

const (
	defaultLeakyReLUSlope = float32(0.01)
	defaultELUAlpha       = float32(1.0)
	defaultSELUAlpha      = float32(1.6732632423543772)
	defaultSELUScale      = float32(1.0507009873554805)
	defaultGELUTanhAlpha  = float32(0.7978845608028654) // sqrt(2/π)
	defaultGELUTanhBeta   = float32(0.044715)
)

func init() {
	registerUnary("sigmoid", func(value float32) float32 {
		return 1 / (1 + float32(math.Exp(float64(-value))))
	})
	registerUnary("silu", func(value float32) float32 {
		return value / (1 + float32(math.Exp(float64(-value))))
	})
	registerUnary("swish", func(value float32) float32 {
		return value / (1 + float32(math.Exp(float64(-value))))
	})
	registerUnary("mish", func(value float32) float32 {
		softplus := math.Log1p(math.Exp(float64(value)))
		return value * float32(math.Tanh(softplus))
	})
	registerUnary("hardsigmoid", func(value float32) float32 {
		x := value/6 + 0.5

		switch {
		case x < 0:
			return 0
		case x > 1:
			return 1
		}

		return x
	})
	registerUnary("hardswish", func(value float32) float32 {
		switch {
		case value <= -3:
			return 0
		case value >= 3:
			return value
		}

		return value * (value + 3) / 6
	})
	registerUnary("softplus", func(value float32) float32 {
		return float32(math.Log1p(math.Exp(float64(value))))
	})
	registerUnary("softsign", func(value float32) float32 {
		return value / (1 + float32(math.Abs(float64(value))))
	})
	registerUnary("elu", func(value float32) float32 {
		if value > 0 {
			return value
		}

		return defaultELUAlpha * (float32(math.Exp(float64(value))) - 1)
	})
	registerUnary("selu", func(value float32) float32 {
		if value > 0 {
			return defaultSELUScale * value
		}

		return defaultSELUScale * defaultSELUAlpha * (float32(math.Exp(float64(value))) - 1)
	})
	registerUnary("leaky_relu", func(value float32) float32 {
		if value > 0 {
			return value
		}

		return defaultLeakyReLUSlope * value
	})
	registerUnary("gelu_tanh", func(value float32) float32 {
		// Approximate GELU per the original BERT paper.
		v := float64(value)
		inner := defaultGELUTanhAlpha * float32(v+float64(defaultGELUTanhBeta)*v*v*v)
		return 0.5 * value * (1 + float32(math.Tanh(float64(inner))))
	})
}
