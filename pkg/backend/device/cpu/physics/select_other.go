//go:build !arm64 && !amd64

package physics

func LaplacianFloat32Native(input, out, scratch []float32, dims []int, invH2 float32) {
	LaplacianFloat32Scalar(input, out, scratch, dims, invH2)
}

func Laplacian4Float32Native(input, out []float32, invDen float32) {
	Laplacian4Float32Scalar(input, out, invDen)
}

func Grad1DFloat32Native(input, out []float32, invTwoDx float32) {
	Grad1DFloat32Scalar(input, out, invTwoDx)
}

func CentralDifferenceInteriorFloat32Native(input, out []float32, invTwoDx float32) {
	CentralDifferenceInteriorFloat32Scalar(input, out, invTwoDx)
}

func QuantumPotentialFloat32Native(
	density, out []float32,
	invH2, scale float32,
) {
	QuantumPotentialFloat32Scalar(density, out, invH2, scale)
}

func MadelungContinuityFloat32Native(
	density, velocity, out []float32,
	invTwoDx float32,
) {
	MadelungContinuityFloat32Scalar(density, velocity, out, invTwoDx)
}
