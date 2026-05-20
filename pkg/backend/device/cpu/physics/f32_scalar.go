package physics

import "math"

func LaplacianFloat32Scalar(input, out, scratch []float32, dims []int, invH2 float32) {
	switch len(dims) {
	case 1:
		laplacian1DScalar(input, out, dims[0], invH2)
	case 2:
		laplacian2DScalar(input, out, dims[0], dims[1], invH2)
	case 3:
		laplacian3DScalar(input, out, dims[0], dims[1], dims[2], invH2)
	}
}

func Laplacian4Float32Scalar(input, out []float32, invDen float32) {
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

func Grad1DFloat32Scalar(input, out []float32, invTwoDx float32) {
	elementCount := len(input)

	for index := 0; index < elementCount; index++ {
		left := input[(index-1+elementCount)%elementCount]
		right := input[(index+1)%elementCount]
		out[index] = (right - left) * invTwoDx
	}
}

func CentralDifferenceInteriorFloat32Scalar(input, out []float32, invTwoDx float32) {
	elementCount := len(input)

	for index := 1; index < elementCount-1; index++ {
		out[index] = (input[index+1] - input[index-1]) * invTwoDx
	}
}

func QuantumPotentialFloat32Scalar(
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

func MadelungContinuityFloat32Scalar(
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

func laplacian1DScalar(input, out []float32, n int, invH2 float32) {
	for index := 0; index < n; index++ {
		left := input[(index-1+n)%n]
		right := input[(index+1)%n]
		out[index] = (left - 2*input[index] + right) * invH2
	}
}

func laplacian2DScalar(input, out []float32, rows, cols int, invH2 float32) {
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			center := input[row*cols+col]
			up := input[((row-1+rows)%rows)*cols+col]
			down := input[((row+1)%rows)*cols+col]
			left := input[row*cols+((col-1+cols)%cols)]
			right := input[row*cols+((col+1)%cols)]
			out[row*cols+col] = (up + down + left + right - 4*center) * invH2
		}
	}
}

func laplacian3DScalar(input, out []float32, depth, rows, cols int, invH2 float32) {
	for depthIndex := 0; depthIndex < depth; depthIndex++ {
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				index := (depthIndex*rows+row)*cols + col
				center := input[index]
				dm := input[(((depthIndex-1+depth)%depth)*rows+row)*cols+col]
				dp := input[(((depthIndex+1)%depth)*rows+row)*cols+col]
				rm := input[(depthIndex*rows+((row-1+rows)%rows))*cols+col]
				rp := input[(depthIndex*rows+((row+1)%rows))*cols+col]
				cm := input[(depthIndex*rows+row)*cols+((col-1+cols)%cols)]
				cp := input[(depthIndex*rows+row)*cols+((col+1)%cols)]
				out[index] = (dm + dp + rm + rp + cm + cp - 6*center) * invH2
			}
		}
	}
}
