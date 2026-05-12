package hawkes

import "math"

// hawkesExpNegBetaDt returns exp(-betaDt) where betaDt = beta * (t - t_i). Called from
// architecture-specific expSum assembly (Plan 9 ABI float arg in platform register).
func hawkesExpNegBetaDt(betaDt float64) float64 {
	return math.Exp(-betaDt)
}
