//go:build arm64

package physics

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/elementwise"
)

func Laplacian1DFloat32Native(input, out []float32, invH2 float32) {
	elementCount := len(input)

	if elementCount == 0 {
		return
	}

	if elementCount == 1 {
		out[0] = 0
		return
	}

	out[0] = (input[elementCount-1] - 2*input[0] + input[1]) * invH2
	interiorCount := elementCount - 2

	if interiorCount > 0 {
		blockCount := interiorCount &^ 3

		if blockCount > 0 {
			Laplacian1DStencilF32NEONAsm(
				&out[1], &input[0], &input[1], &input[2],
				invH2, blockCount,
			)
		}

		for index := 1 + blockCount; index < elementCount-1; index++ {
			left := input[index-1]
			center := input[index]
			right := input[index+1]
			out[index] = (left - 2*center + right) * invH2
		}
	}

	lastIndex := elementCount - 1
	out[lastIndex] = (input[lastIndex-1] - 2*input[lastIndex] + input[0]) * invH2
}

func Laplacian2DFloat32Native(input, out, scratch []float32, rows, cols int, invH2 float32) {
	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowOffset := rowIndex * cols
		Laplacian1DFloat32Native(input[rowOffset:rowOffset+cols], out[rowOffset:rowOffset+cols], invH2)
	}

	columnScratch := scratch[rows : rows*2]

	for colIndex := 0; colIndex < cols; colIndex++ {
		for rowIndex := 0; rowIndex < rows; rowIndex++ {
			scratch[rowIndex] = input[rowIndex*cols+colIndex]
		}

		Laplacian1DFloat32Native(scratch[:rows], columnScratch, invH2)

		for rowIndex := 0; rowIndex < rows; rowIndex++ {
			out[rowIndex*cols+colIndex] += columnScratch[rowIndex]
		}
	}
}

func Laplacian3DFloat32Native(
	input, out, scratch []float32,
	depth, rows, cols int,
	invH2 float32,
) {
	for depthIndex := 0; depthIndex < depth; depthIndex++ {
		for rowIndex := 0; rowIndex < rows; rowIndex++ {
			rowOffset := (depthIndex*rows + rowIndex) * cols
			Laplacian1DFloat32Native(
				input[rowOffset:rowOffset+cols],
				out[rowOffset:rowOffset+cols],
				invH2,
			)
		}
	}

	columnScratch := scratch[rows : rows*2]

	for depthIndex := 0; depthIndex < depth; depthIndex++ {
		for colIndex := 0; colIndex < cols; colIndex++ {
			for rowIndex := 0; rowIndex < rows; rowIndex++ {
				scratch[rowIndex] = input[(depthIndex*rows+rowIndex)*cols+colIndex]
			}

			Laplacian1DFloat32Native(scratch[:rows], columnScratch, invH2)

			for rowIndex := 0; rowIndex < rows; rowIndex++ {
				out[(depthIndex*rows+rowIndex)*cols+colIndex] += columnScratch[rowIndex]
			}
		}
	}

	depthScratch := scratch[depth : depth*2]

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for colIndex := 0; colIndex < cols; colIndex++ {
			for depthIndex := 0; depthIndex < depth; depthIndex++ {
				scratch[depthIndex] = input[(depthIndex*rows+rowIndex)*cols+colIndex]
			}

			Laplacian1DFloat32Native(scratch[:depth], depthScratch, invH2)

			for depthIndex := 0; depthIndex < depth; depthIndex++ {
				out[(depthIndex*rows+rowIndex)*cols+colIndex] += depthScratch[depthIndex]
			}
		}
	}
}

func Laplacian4Float32Native(input, out []float32, invDen float32) {
	elementCount := len(input)

	if elementCount == 0 {
		return
	}

	if elementCount < 5 {
		for index := 0; index < elementCount; index++ {
			um2 := input[(index-2+elementCount)%elementCount]
			um1 := input[(index-1+elementCount)%elementCount]
			u0 := input[index]
			up1 := input[(index+1)%elementCount]
			up2 := input[(index+2)%elementCount]
			out[index] = (-um2 + 16*um1 - 30*u0 + 16*up1 - up2) * invDen
		}

		return
	}

	for index := 0; index < 2; index++ {
		um2 := input[(index-2+elementCount)%elementCount]
		um1 := input[(index-1+elementCount)%elementCount]
		u0 := input[index]
		up1 := input[(index+1)%elementCount]
		up2 := input[(index+2)%elementCount]
		out[index] = (-um2 + 16*um1 - 30*u0 + 16*up1 - up2) * invDen
	}

	interiorCount := elementCount - 4
	blockCount := interiorCount &^ 3

	if blockCount > 0 {
		baseIndex := 2
		Laplacian4StencilF32NEONAsm(
			&out[baseIndex],
			&input[baseIndex-2],
			&input[baseIndex-1],
			&input[baseIndex],
			&input[baseIndex+1],
			&input[baseIndex+2],
			invDen,
			blockCount,
		)
	}

	for index := 2 + blockCount; index < elementCount-2; index++ {
		um2 := input[index-2]
		um1 := input[index-1]
		u0 := input[index]
		up1 := input[index+1]
		up2 := input[index+2]
		out[index] = (-um2 + 16*um1 - 30*u0 + 16*up1 - up2) * invDen
	}

	for index := elementCount - 2; index < elementCount; index++ {
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

	if elementCount == 0 {
		return
	}

	if elementCount == 1 {
		out[0] = 0
		return
	}

	out[0] = (input[1] - input[elementCount-1]) * invTwoDx
	interiorCount := elementCount - 2

	if interiorCount > 0 {
		blockCount := interiorCount &^ 3

		if blockCount > 0 {
			Grad1DStencilF32NEONAsm(
				&out[1], &input[0], &input[2],
				invTwoDx, blockCount,
			)
		}

		for index := 1 + blockCount; index < elementCount-1; index++ {
			out[index] = (input[index+1] - input[index-1]) * invTwoDx
		}
	}

	lastIndex := elementCount - 1
	out[lastIndex] = (input[0] - input[lastIndex-1]) * invTwoDx
}

func CentralDifferenceInteriorFloat32Native(input, out []float32, invTwoDx float32) {
	elementCount := len(input)

	if elementCount <= 2 {
		return
	}

	interiorCount := elementCount - 2
	blockCount := interiorCount &^ 3

	if blockCount > 0 {
		Grad1DStencilF32NEONAsm(
			&out[1], &input[0], &input[2],
			invTwoDx, blockCount,
		)
	}

	for index := 1 + blockCount; index < elementCount-1; index++ {
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
	sqrtRho := BorrowFloat32Buffer(elementCount)
	laplacian := BorrowFloat32Buffer(elementCount)

	defer ReleaseFloat32Buffer(sqrtRho)
	defer ReleaseFloat32Buffer(laplacian)

	for index, value := range density {
		if value <= eps {
			sqrtRho[index] = float32(math.Sqrt(float64(eps)))
			continue
		}

		sqrtRho[index] = float32(math.Sqrt(float64(value)))
	}

	interiorCount := elementCount - 2
	blockCount := interiorCount &^ 3

	if blockCount > 0 {
		Laplacian1DStencilF32NEONAsm(
			&laplacian[1],
			&sqrtRho[0],
			&sqrtRho[1],
			&sqrtRho[2],
			invH2,
			blockCount,
		)
	}

	for index := 1 + blockCount; index < elementCount-1; index++ {
		laplacian[index] = (sqrtRho[index+1] - 2*sqrtRho[index] + sqrtRho[index-1]) * invH2
	}

	for index := 1; index < elementCount-1; index++ {
		if density[index] <= eps {
			out[index] = 0
			continue
		}

		out[index] = scale * laplacian[index] / sqrtRho[index]
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

	flux := BorrowFloat32Buffer(elementCount)

	defer ReleaseFloat32Buffer(flux)

	elementwise.MulFloat32Native(flux, density, velocity)
	CentralDifferenceInteriorFloat32Native(flux, out, invTwoDx)
}

func LaplacianFloat32Native(input, out, scratch []float32, dims []int, invH2 float32) {
	switch len(dims) {
	case 1:
		Laplacian1DFloat32Native(input, out, invH2)
	case 2:
		Laplacian2DFloat32Native(input, out, scratch, dims[0], dims[1], invH2)
	case 3:
		Laplacian3DFloat32Native(input, out, scratch, dims[0], dims[1], dims[2], invH2)
	}
}
