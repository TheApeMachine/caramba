//go:build !arm64

package cpu

import "math"

func LaplacianFloat32Native(input, out, scratch []float32, dims []int, invH2 float32) {
	switch len(dims) {
	case 1:
		laplacian1D(input, out, dims[0], invH2)
	case 2:
		laplacian2D(input, out, dims[0], dims[1], invH2)
	case 3:
		laplacian3D(input, out, dims[0], dims[1], dims[2], invH2)
	}
}

func Laplacian4Float32Native(input, out []float32, invDen float32) {
	elementCount := len(input)

	for index := 0; index < elementCount; index++ {
		um2 := input[(index-2+elementCount)%elementCount]
		um1 := input[(index-1+elementCount)%elementCount]
		u0 := input[index]
		up1 := input[(index+1)%elementCount]
		up2 := input[(index+2)%elementCount]
		out[index] = (-um2 + 16*um1 - 30*u0 + 16*up1 - up2) * invDen
	}
}

func Grad1DFloat32Native(input, out []float32, invTwoDx float32) {
	elementCount := len(input)

	for index := 0; index < elementCount; index++ {
		left := input[(index-1+elementCount)%elementCount]
		right := input[(index+1)%elementCount]
		out[index] = (right - left) * invTwoDx
	}
}

func CentralDifferenceInteriorFloat32Native(input, out []float32, invTwoDx float32) {
	elementCount := len(input)

	for index := 1; index < elementCount-1; index++ {
		out[index] = (input[index+1] - input[index-1]) * invTwoDx
	}
}

func QuantumPotentialFloat32Native(
	density, out []float32,
	invH2, scale float32,
) {
	elementCount := len(density)

	if elementCount == 0 {
		return
	}

	out[0] = 0
	out[elementCount-1] = 0

	if elementCount <= 2 {
		return
	}

	const eps = float32(1e-12)

	for index := 1; index < elementCount-1; index++ {
		rho := float64(density[index])

		if rho <= float64(eps) {
			out[index] = 0
			continue
		}

		sqrtRho := math.Sqrt(rho)
		sqrtLeft := math.Sqrt(math.Max(float64(eps), float64(density[index-1])))
		sqrtRight := math.Sqrt(math.Max(float64(eps), float64(density[index+1])))
		laplacian := (sqrtRight - 2*sqrtRho + sqrtLeft) * float64(invH2)
		out[index] = float32(float64(scale) * laplacian / sqrtRho)
	}
}

func MadelungContinuityFloat32Native(
	density, velocity, out []float32,
	invTwoDx float32,
) {
	elementCount := len(density)

	if elementCount == 0 {
		return
	}

	out[0] = 0
	out[elementCount-1] = 0

	if elementCount <= 2 {
		return
	}

	for index := 1; index < elementCount-1; index++ {
		fluxRight := density[index+1] * velocity[index+1]
		fluxLeft := density[index-1] * velocity[index-1]
		out[index] = (fluxRight - fluxLeft) * invTwoDx
	}
}
