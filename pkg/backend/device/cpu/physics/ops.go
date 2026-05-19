package physics

import "unsafe"

func Laplacian(
	input, output unsafe.Pointer,
	dims []int,
	spacing float32,
) {
	elementCount := denseElementCount(dims)

	if elementCount == 0 {
		return
	}

	inputView := unsafe.Slice((*float32)(input), elementCount)
	outputView := unsafe.Slice((*float32)(output), elementCount)
	dxValue := float64(spacing)

	if dxValue <= 0 {
		dxValue = 1.0
	}

	dxSquared := float32(dxValue * dxValue)
	invH2 := float32(1.0 / float64(dxSquared))

	switch len(dims) {
	case 1:
		LaplacianFloat32Native(inputView, outputView, nil, dims, invH2)
	case 2, 3:
		scratchAxis := dims[0]

		if len(dims) >= 2 && dims[1] > scratchAxis {
			scratchAxis = dims[1]
		}

		scratch := make([]float32, scratchAxis*2)
		LaplacianFloat32Native(inputView, outputView, scratch, dims, invH2)
	}
}

func Laplacian4(input, output unsafe.Pointer, count int, spacing float32) {
	if count == 0 {
		return
	}

	inputView := unsafe.Slice((*float32)(input), count)
	outputView := unsafe.Slice((*float32)(output), count)
	dxValue := float64(spacing)

	if dxValue <= 0 {
		dxValue = 1.0
	}

	dxSquared := float32(dxValue * dxValue)
	denominator := 12 * dxSquared
	invDen := float32(1.0 / float64(denominator))

	Laplacian4Float32Native(inputView, outputView, invDen)
}

func Grad1D(input, output unsafe.Pointer, count int, spacing float32) {
	if count == 0 {
		return
	}

	inputView := unsafe.Slice((*float32)(input), count)
	outputView := unsafe.Slice((*float32)(output), count)
	dxValue := float64(spacing)

	if dxValue <= 0 {
		dxValue = 1.0
	}

	denominator := float32(2 * dxValue)
	invTwoDx := float32(1.0 / float64(denominator))

	Grad1DFloat32Native(inputView, outputView, invTwoDx)
}

func Divergence1D(input, output unsafe.Pointer, count int, spacing float32) {
	Grad1D(input, output, count, spacing)
}

func FFT1D(
	realIn, imagIn, realOut, imagOut unsafe.Pointer,
	count int,
) {
	if count == 0 {
		return
	}

	fftFloat32(
		unsafe.Slice((*float32)(realIn), count),
		unsafe.Slice((*float32)(imagIn), count),
		unsafe.Slice((*float32)(realOut), count),
		unsafe.Slice((*float32)(imagOut), count),
		false,
	)
}

func IFFT1D(
	realIn, imagIn, realOut, imagOut unsafe.Pointer,
	count int,
) {
	if count == 0 {
		return
	}

	fftFloat32(
		unsafe.Slice((*float32)(realIn), count),
		unsafe.Slice((*float32)(imagIn), count),
		unsafe.Slice((*float32)(realOut), count),
		unsafe.Slice((*float32)(imagOut), count),
		true,
	)
}

func QuantumPotential(
	density, output unsafe.Pointer,
	count int,
	spacing float32,
) {
	if count == 0 {
		return
	}

	densityView := unsafe.Slice((*float32)(density), count)
	outputView := unsafe.Slice((*float32)(output), count)
	dxValue := float64(spacing)

	if dxValue <= 0 {
		dxValue = 1.0
	}

	invH2 := float32(1.0 / (dxValue * dxValue))
	scale := float32(-float64(defaultReducedPlanck*defaultReducedPlanck) / (2 * float64(defaultMass)))

	QuantumPotentialFloat32Native(densityView, outputView, invH2, scale)
}

func BohmianVelocity(
	phase, output unsafe.Pointer,
	count int,
	spacing float32,
) {
	if count == 0 {
		return
	}

	phaseView := unsafe.Slice((*float32)(phase), count)
	outputView := unsafe.Slice((*float32)(output), count)
	dxValue := float64(spacing)

	if dxValue <= 0 {
		dxValue = 1.0
	}

	inverseDoubleDx := float32(1.0 / (2 * dxValue * float64(defaultMass)))

	outputView[0] = 0
	outputView[count-1] = 0

	CentralDifferenceInteriorFloat32Native(phaseView, outputView, inverseDoubleDx)
}

func MadelungContinuity(
	density, velocity, residual unsafe.Pointer,
	count int,
	spacing float32,
) {
	if count == 0 {
		return
	}

	densityView := unsafe.Slice((*float32)(density), count)
	velocityView := unsafe.Slice((*float32)(velocity), count)
	residualView := unsafe.Slice((*float32)(residual), count)
	dxValue := float64(spacing)

	if dxValue <= 0 {
		dxValue = 1.0
	}

	inverseDoubleDx := float32(1.0 / (2 * dxValue))

	MadelungContinuityFloat32Native(densityView, velocityView, residualView, inverseDoubleDx)
}

func denseElementCount(dims []int) int {
	elementCount := 1

	for _, dim := range dims {
		elementCount *= dim
	}

	return elementCount
}
