package math

/*
ExpVec writes exp(input[i]) into output[i].
*/
func ExpVec(output, input []float64) {
	expVec(output, input)
}

/*
LogVec writes log(input[i]) into output[i].
*/
func LogVec(output, input []float64) {
	logVec(output, input)
}

/*
MulVec writes left[i] * right[i] into output[i].
*/
func MulVec(output, left, right []float64) {
	mulVec(output, left, right)
}

/*
MatMul writes left[M*K] x right[K*N] into output[M*N].
*/
func MatMul(output, left, right []float64, rows, inner, cols int) {
	matmulKernel(output, left, right, rows, inner, cols)
}

/*
AddScaledVec accumulates output[i] += scale * input[i].
*/
func AddScaledVec(output, input []float64, scale float64) {
	addScaledVec(output, input, scale)
}

/*
AddScalarVec accumulates output[i] += scalar.
*/
func AddScalarVec(output []float64, scalar float64) {
	addScalarVec(output, scalar)
}

/*
ScaleVec multiplies output in place by scalar.
*/
func ScaleVec(output []float64, scalar float64) {
	mulScalar(output, scalar)
}

/*
ClampVec clamps output in place to [low, high].
*/
func ClampVec(output []float64, low, high float64) {
	clampVec(output, low, high)
}

/*
ReduceSum returns the sum of input.
*/
func ReduceSum(input []float64) float64 {
	return reduceSum(input)
}
