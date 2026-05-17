package compose

import (
	"fmt"
	"math"
)

/*
InputSpec describes a graph-level input the compiled topology needs
to declare. Kind is an optional hint that helps patterns pick the
right entry node, such as "tokens" for an embedding lookup.
*/
type InputSpec struct {
	Name string
	Kind string
}

/*
Hints carry the small amount of context that cannot be inferred from
the safetensors file alone: graph inputs, final output, and optional
numeric metadata consumed by patterns.
*/
type Hints struct {
	Inputs   []InputSpec
	Output   string
	Metadata map[string]any
}

const (
	/*
		HintLayerNormEpsilon overrides the math.layernorm eps config emitted
		by compose patterns. The default remains 1e-5 when unset.
	*/
	HintLayerNormEpsilon = "layernorm_epsilon"

	/*
		HintRMSNormEpsilon overrides the math.rmsnorm eps config emitted by
		compose patterns. The default remains 1e-6 when unset; Llama-2-style
		callers can set rmsnorm_epsilon to 1e-5.
	*/
	HintRMSNormEpsilon = "rmsnorm_epsilon"
)

/*
GetFloat returns a positive float hint value or fallback when key is
absent.
*/
func (hints Hints) GetFloat(key string, fallback float64) (float64, error) {
	raw, ok := hints.Metadata[key]

	if !ok {
		return fallback, nil
	}

	value, ok := hintFloat(raw)

	if !ok {
		return 0, fmt.Errorf("compose: hint %s must be numeric, got %T", key, raw)
	}

	if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("compose: hint %s must be > 0, got %g", key, value)
	}

	return value, nil
}

func hintFloat(raw any) (float64, bool) {
	switch value := raw.(type) {
	case float64:
		return value, true
	case float32:
		return float64(value), true
	case int:
		return float64(value), true
	case int64:
		return float64(value), true
	}

	return 0, false
}
