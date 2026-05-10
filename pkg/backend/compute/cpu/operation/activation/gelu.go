package activation

/*
Gelu evaluates the standard approximate GELU map using SIMD instructions on
amd64/arm64 and a scalar fallback on other platforms.

The formulation matches the assembly implementations: tanh uses the rational
approximation z·(27+z²)/(27+9z²).
*/
type Gelu struct{}

/*
NewGelu returns an Operation that computes approximate GELU elementwise.
*/
func NewGelu() *Gelu {
	return &Gelu{}
}

/*
Forward maps each input element through the approximate GELU formulation and
returns freshly allocated outputs.
*/
func (gelu *Gelu) Forward(shape []int, data ...[]float64) []float64 {
	input := data[0]
	out := make([]float64, len(input))
	applyGeLU(out, input)
	return out
}
