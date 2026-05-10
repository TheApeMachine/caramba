package math

// Exported SIMD-dispatched vector primitives for use by sibling packages
// (optimizers, etc.) that cannot call unexported functions across package boundaries.

// AddScaledVec: dst[i] += scale * src[i]  (AXPY)
func AddScaledVec(dst, src []float64, scale float64) { addScaledVec(dst, src, scale) }

// AddVec: dst[i] += src[i]
func AddVec(dst, src []float64) { addVec(dst, dst, src) }

// MulScalar: dst[i] *= s
func MulScalar(dst []float64, s float64) { mulScalar(dst, s) }

// SqrtVec: dst[i] = sqrt(src[i])
func SqrtVec(dst, src []float64) { sqrtVec(dst, src) }

// AddScalarVec: dst[i] += scalar
func AddScalarVec(dst []float64, scalar float64) { addScalarVec(dst, scalar) }

// DivVec: dst[i] = a[i] / b[i]
func DivVec(dst, a, b []float64) { divVec(dst, a, b) }

// L2NormSq: returns sum(a[i]^2)
func L2NormSq(a []float64) float64 { return l2NormSq(a) }

// ClampVec: dst[i] = clamp(dst[i], lo, hi)
func ClampVec(dst []float64, lo, hi float64) { clampVec(dst, lo, hi) }

// MulVec: dst[i] = a[i] * b[i]
func MulVec(dst, a, b []float64) { mulVec(dst, a, b) }

// ScaleVec: dst[i] *= s  (alias for MulScalar)
func ScaleVec(dst []float64, s float64) { mulScalar(dst, s) }
