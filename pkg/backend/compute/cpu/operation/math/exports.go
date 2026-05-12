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

// DivVec: dst[i] = a[i] / b[i]; follows IEEE-754 — b[i]==0 yields ±Inf or NaN, no panic.
func DivVec(dst, a, b []float64) { divVec(dst, a, b) }

// L2NormSq: returns sum(a[i]^2)
func L2NormSq(a []float64) float64 { return l2NormSq(a) }

// ClampVec: dst[i] = clamp(dst[i], lo, hi)
func ClampVec(dst []float64, lo, hi float64) { clampVec(dst, lo, hi) }

// MulVec: dst[i] = a[i] * b[i]
func MulVec(dst, a, b []float64) { mulVec(dst, a, b) }

// ScaleVec: dst[i] *= s  (alias for MulScalar)
func ScaleVec(dst []float64, s float64) { mulScalar(dst, s) }

// MatMul: C = A [M×K] · B [K×N], row-major, dispatched to AVX2/NEON/SSE2.
// Requires len(dst) >= M*N, len(a) >= M*K, len(b) >= K*N; violations panic with out-of-bounds access.
func MatMul(dst, a, b []float64, M, K, N int) { applyMatMul(dst, a, b, M, K, N) }

// SoftmaxSlice: in-place softmax on a single flat row. No-op for empty slice.
// Numerically stable via max subtraction before exp.
func SoftmaxSlice(row []float64) {
	if len(row) == 0 {
		return
	}
	softmaxRow(row)
}

// ExpVec: dst[i] = exp(src[i]) via vectorized polynomial approximation.
func ExpVec(dst, src []float64) { expVec(dst, src) }

// LogVec: dst[i] = log(src[i]) for src[i] > 0.
func LogVec(dst, src []float64) { logVec(dst, src) }

// ReduceSum: returns sum of all elements in a.
func ReduceSum(a []float64) float64 { return reduceSum(a) }

// ReduceMax: returns max of a; returns -math.MaxFloat64 for empty input.
func ReduceMax(a []float64) float64 { return reduceMax(a) }

// SignVec: dst[i] = sign(src[i])  (-1, 0, +1).
func SignVec(dst, src []float64) { signVec(dst, src) }
